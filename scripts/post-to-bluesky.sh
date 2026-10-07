#!/usr/bin/env bash
#
# Announce a blog article on Bluesky.
#
# Usage:
#   scripts/post-to-bluesky.sh [--dry-run] <path/to/post.md>
#
# Required environment variables (not needed with --dry-run):
#   BLUESKY_HANDLE        Your Bluesky handle, e.g. name.bsky.social
#   BLUESKY_APP_PASSWORD  An app password created in Bluesky settings
#
# Optional:
#   SITE_BASE_URL         Site root (default: https://bergstrom.org)
#
set -euo pipefail

SITE_BASE_URL="${SITE_BASE_URL:-https://bergstrom.org}"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

POST_FILE="${1:-}"
if [[ -z "$POST_FILE" || ! -f "$POST_FILE" ]]; then
  echo "error: post file not found: '${POST_FILE:-}'" >&2
  exit 1
fi

# --- read front matter (the first block delimited by lines of ---) -----------
front_matter="$(awk 'BEGIN{n=0} /^---[[:space:]]*$/{n++; if(n==2) exit; next} n==1{print}' "$POST_FILE")"
body="$(awk 'BEGIN{n=0} /^---[[:space:]]*$/{n++; next} n>=2{print}' "$POST_FILE")"

get_fm() { # $1 = key -> value (quotes stripped)
  printf '%s\n' "$front_matter" \
    | sed -n "s/^$1:[[:space:]]*//p" \
    | head -n1 \
    | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//"
}

TITLE="$(get_fm title)"
DESCRIPTION="$(get_fm description)"
SLUG_OVERRIDE="$(get_fm slug)"
BLUESKY_IMAGE="$(get_fm bluesky_image)"

if [[ -z "$TITLE" ]]; then
  echo "error: no 'title' in front matter of $POST_FILE" >&2
  exit 1
fi

# --- resolve the optional link-card thumbnail --------------------------------
# `bluesky_image` is a path to an image in static/, e.g. /my_cover.webp. It is
# attached as the external link card's `thumb`. Omitting it keeps the old
# image-less behavior.
IMAGE_PATH=""
IMAGE_MIME=""
if [[ -n "$BLUESKY_IMAGE" ]]; then
  if [[ -f "$BLUESKY_IMAGE" ]]; then
    candidate="$BLUESKY_IMAGE"
  elif [[ "$BLUESKY_IMAGE" == /* ]]; then
    candidate="static${BLUESKY_IMAGE}"
  else
    candidate="static/${BLUESKY_IMAGE}"
  fi

  if [[ ! -f "$candidate" ]]; then
    echo "error: bluesky_image not found: '$BLUESKY_IMAGE' (looked for '$candidate')" >&2
    exit 1
  fi
  IMAGE_PATH="$candidate"

  image_ext="$(printf '%s' "${IMAGE_PATH##*.}" | tr '[:upper:]' '[:lower:]')"
  case "$image_ext" in
    jpg|jpeg) IMAGE_MIME="image/jpeg" ;;
    png)      IMAGE_MIME="image/png" ;;
    webp)     IMAGE_MIME="image/webp" ;;
    gif)      IMAGE_MIME="image/gif" ;;
    *)
      echo "error: unsupported bluesky_image type: .$image_ext" >&2
      exit 1
      ;;
  esac

  image_size="$(wc -c < "$IMAGE_PATH" | tr -d '[:space:]')"
  MAX_IMAGE_BYTES=1000000
  if (( image_size > MAX_IMAGE_BYTES )); then
    echo "error: bluesky_image is ${image_size} bytes; Bluesky limits thumbnails to ${MAX_IMAGE_BYTES} bytes" >&2
    exit 1
  fi
fi

# --- derive the article URL --------------------------------------------------
# Hugo lowercases URL paths by default; no post in this repo sets a custom slug.
if [[ -n "$SLUG_OVERRIDE" ]]; then
  SLUG="$SLUG_OVERRIDE"
else
  SLUG="$(basename "$POST_FILE" .md)"
fi
SLUG="$(printf '%s' "$SLUG" | tr '[:upper:]' '[:lower:]')"
POST_URL="${SITE_BASE_URL%/}/posts/${SLUG}/"

# --- fall back to the first paragraph if there is no description -------------
if [[ -z "$DESCRIPTION" ]]; then
  DESCRIPTION="$(printf '%s\n' "$body" | awk '
    /^[[:space:]]*$/ { if (started) exit; next }
    /^#/            { next }
    /^\$\$/         { next }
    /^```/          { next }
    { started=1; print }
  ' \
    | tr '\n' ' ' \
    | sed -E 's/\[([^]]*)\]\([^)]*\)/\1/g' \
    | sed -E 's/[*_`]//g' \
    | sed -E 's/[[:space:]]+/ /g' \
    | sed -E 's/^ //')"
fi
if [[ ${#DESCRIPTION} -gt 200 ]]; then
  DESCRIPTION="$(printf '%s' "${DESCRIPTION:0:200}" | sed -E 's/[[:space:]][^[:space:]]*$//')…"
fi

# --- compose the post text (Bluesky allows 300 characters) -------------------
PREFIX="New on the blog — "
TEXT="${PREFIX}${TITLE}

${POST_URL}"
if [[ ${#TEXT} -gt 300 ]]; then
  max_title=$(( 300 - ${#TEXT} + ${#TITLE} ))
  if [[ $max_title -lt 10 ]]; then max_title=10; fi
  TITLE_SHORT="$(printf '%s' "${TITLE:0:max_title}" | sed -E 's/[[:space:]][^[:space:]]*$//')…"
  TEXT="${PREFIX}${TITLE_SHORT}

${POST_URL}"
fi

CREATED_AT="$(date -u +%Y-%m-%dT%H:%M:%S.000Z)"

echo "Post text:"
echo "---"
printf '%s\n' "$TEXT"
echo "---"
echo "link card URL:  $POST_URL"
echo "link card title: $TITLE"
echo "link card desc:  $DESCRIPTION"
if [[ -n "$IMAGE_PATH" ]]; then
  echo "link card thumb: $IMAGE_PATH ($IMAGE_MIME)"
else
  echo "link card thumb: (none)"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "(dry run — nothing was sent)"
  exit 0
fi

: "${BLUESKY_HANDLE:?BLUESKY_HANDLE is not set}"
: "${BLUESKY_APP_PASSWORD:?BLUESKY_APP_PASSWORD is not set}"

# Be forgiving about common copy/paste slips: an accidental leading "@" on the
# handle, or stray whitespace around either value.
BLUESKY_HANDLE="${BLUESKY_HANDLE#@}"
BLUESKY_HANDLE="$(printf '%s' "$BLUESKY_HANDLE" | tr -d '[:space:]')"
BLUESKY_APP_PASSWORD="$(printf '%s' "$BLUESKY_APP_PASSWORD" | tr -d '[:space:]')"

# --------------------------------------------------------------------------
# HTTP helper — retries transient failures and, when a response is not the
# JSON we expect, prints the raw body instead of dying with an opaque
# `jq: parse error`. (A single non-JSON response aborted the whole run and,
# under the old logic, the post was then considered "already announced".)
# --------------------------------------------------------------------------
HTTP_ATTEMPTS="${HTTP_ATTEMPTS:-4}"
BODY_FILE="$(mktemp)"
trap 'rm -f "$BODY_FILE"' EXIT

api_post() { # usage: api_post <url> <curl args...>
  local url="$1"; shift
  local attempt=1 code=""
  while (( attempt <= HTTP_ATTEMPTS )); do
    code="$(curl -sS --max-time 90 -o "$BODY_FILE" -w '%{http_code}' "$@" "$url" 2>/dev/null || true)"
    if [[ "$code" == 2* ]]; then
      return 0
    fi
    echo "warning: POST ${url##*/} -> HTTP ${code:-no-response} (attempt $attempt/$HTTP_ATTEMPTS)" >&2
    if (( attempt < HTTP_ATTEMPTS )); then sleep $(( attempt * 3 )); fi
    attempt=$(( attempt + 1 ))
  done
  echo "error: request failed after $HTTP_ATTEMPTS attempts: $url" >&2
  echo "last response body:" >&2
  cat "$BODY_FILE" >&2
  echo >&2
  return 1
}

require_json() { # reads stdin; fails loudly (printing the body) if it isn't JSON
  local body
  body="$(cat)"
  if ! printf '%s' "$body" | jq -e . >/dev/null 2>&1; then
    echo "error: expected a JSON response, got:" >&2
    printf '%s\n' "$body" >&2
    return 1
  fi
  printf '%s' "$body"
}

# --- log in and obtain a session token ---------------------------------------
api_post "https://bsky.social/xrpc/com.atproto.server.createSession" \
  -H "Content-Type: application/json" \
  --data "$(jq -n --arg id "$BLUESKY_HANDLE" --arg pw "$BLUESKY_APP_PASSWORD" \
    '{identifier:$id, password:$pw}')"
session="$(require_json < "$BODY_FILE")"

access_jwt="$(printf '%s' "$session" | jq -r '.accessJwt // empty')"
did="$(printf '%s' "$session" | jq -r '.did // empty')"
if [[ -z "$access_jwt" || -z "$did" ]]; then
  echo "error: Bluesky login failed:" >&2
  printf '%s\n' "$session" | jq . >&2 2>/dev/null || printf '%s\n' "$session" >&2
  exit 1
fi

# --- upload the optional thumbnail blob --------------------------------------
thumb_json="null"
if [[ -n "$IMAGE_PATH" ]]; then
  api_post "https://bsky.social/xrpc/com.atproto.repo.uploadBlob" \
    -H "Authorization: Bearer $access_jwt" \
    -H "Content-Type: $IMAGE_MIME" \
    --data-binary "@$IMAGE_PATH"
  upload_result="$(require_json < "$BODY_FILE")"

  thumb_json="$(printf '%s' "$upload_result" | jq -c '.blob // empty')"
  if [[ -z "$thumb_json" ]]; then
    echo "error: Bluesky image upload failed:" >&2
    printf '%s\n' "$upload_result" | jq . >&2 2>/dev/null || printf '%s\n' "$upload_result" >&2
    exit 1
  fi
fi

# --- create the post, with a link preview card -------------------------------
payload="$(jq -n \
  --arg repo "$did" \
  --arg text "$TEXT" \
  --arg createdAt "$CREATED_AT" \
  --arg uri "$POST_URL" \
  --arg title "$TITLE" \
  --arg description "$DESCRIPTION" \
  --argjson thumb "$thumb_json" \
  '{
     repo: $repo,
     collection: "app.bsky.feed.post",
     record: {
       "$type": "app.bsky.feed.post",
       text: $text,
       createdAt: $createdAt,
       embed: {
         "$type": "app.bsky.embed.external",
         external: (
           { uri: $uri, title: $title, description: $description }
           + (if $thumb == null then {} else { thumb: $thumb } end)
         )
       }
     }
   }')"

api_post "https://bsky.social/xrpc/com.atproto.repo.createRecord" \
  -H "Authorization: Bearer $access_jwt" \
  -H "Content-Type: application/json" \
  --data "$payload"
result="$(require_json < "$BODY_FILE")"

uri="$(printf '%s' "$result" | jq -r '.uri // empty')"
if [[ -z "$uri" ]]; then
  echo "error: Bluesky post failed:" >&2
  printf '%s\n' "$result" | jq . >&2 2>/dev/null || printf '%s\n' "$result" >&2
  exit 1
fi

echo "Posted: $uri"
