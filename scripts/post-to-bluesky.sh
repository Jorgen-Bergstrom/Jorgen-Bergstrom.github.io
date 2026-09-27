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

if [[ -z "$TITLE" ]]; then
  echo "error: no 'title' in front matter of $POST_FILE" >&2
  exit 1
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

# --- log in and obtain a session token ---------------------------------------
session="$(curl -sS -X POST \
  "https://bsky.social/xrpc/com.atproto.server.createSession" \
  -H "Content-Type: application/json" \
  --data "$(jq -n --arg id "$BLUESKY_HANDLE" --arg pw "$BLUESKY_APP_PASSWORD" \
    '{identifier:$id, password:$pw}')")"

access_jwt="$(printf '%s' "$session" | jq -r '.accessJwt // empty')"
did="$(printf '%s' "$session" | jq -r '.did // empty')"
if [[ -z "$access_jwt" || -z "$did" ]]; then
  echo "error: Bluesky login failed:" >&2
  printf '%s\n' "$session" | jq . >&2 2>/dev/null || printf '%s\n' "$session" >&2
  exit 1
fi

# --- create the post, with a link preview card -------------------------------
payload="$(jq -n \
  --arg repo "$did" \
  --arg text "$TEXT" \
  --arg createdAt "$CREATED_AT" \
  --arg uri "$POST_URL" \
  --arg title "$TITLE" \
  --arg description "$DESCRIPTION" \
  '{
     repo: $repo,
     collection: "app.bsky.feed.post",
     record: {
       "$type": "app.bsky.feed.post",
       text: $text,
       createdAt: $createdAt,
       embed: {
         "$type": "app.bsky.embed.external",
         external: {
           uri: $uri,
           title: $title,
           description: $description
         }
       }
     }
   }')"

result="$(curl -sS -X POST \
  "https://bsky.social/xrpc/com.atproto.repo.createRecord" \
  -H "Authorization: Bearer $access_jwt" \
  -H "Content-Type: application/json" \
  --data "$payload")"

uri="$(printf '%s' "$result" | jq -r '.uri // empty')"
if [[ -z "$uri" ]]; then
  echo "error: Bluesky post failed:" >&2
  printf '%s\n' "$result" | jq . >&2 2>/dev/null || printf '%s\n' "$result" >&2
  exit 1
fi

echo "Posted: $uri"
