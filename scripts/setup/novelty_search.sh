#!/usr/bin/env bash
# arXiv novelty search for AdaScale-CLIP
set -e
OUT="$1"
mkdir -p "$(dirname "$OUT")"
> "$OUT"

queries=(
  'all:%22early%20exit%22%20AND%20all:%22CLIP%22'
  'all:%22multi-scale%22%20AND%20all:%22zero-shot%20classification%22'
  'all:%22adaptive%22%20AND%20all:%22multi-crop%22%20AND%20all:%22CLIP%22'
  'all:%22dynamic%20resolution%22%20AND%20all:%22vision-language%22'
  'all:%22compute-adaptive%22%20AND%20all:%22inference%22%20AND%20all:%22vision%22'
  'all:%22cascade%22%20AND%20all:%22CLIP%22%20AND%20all:%22zero-shot%22'
  'all:%22test-time%20efficiency%22%20AND%20all:%22CLIP%22'
  'all:%22visual%20prototype%22%20AND%20all:%22diffusion%22%20AND%20all:%22zero-shot%22'
  'all:%22confidence%22%20AND%20all:%22multi-scale%22%20AND%20all:%22classification%22'
  'all:%22anytime%22%20AND%20all:%22inference%22%20AND%20all:%22CLIP%22'
  'all:%22sample-adaptive%22%20AND%20all:%22computation%22'
  'all:%22stability%22%20AND%20all:%22early%20termination%22%20AND%20all:%22classification%22'
)

for q in "${queries[@]}"; do
  echo "=================== QUERY: $q ===================" >> "$OUT"
  curl -s --max-time 20 "https://export.arxiv.org/api/query?search_query=${q}&start=0&max_results=15&sortBy=relevance" \
    | python3 -c "
import sys, re
data = sys.stdin.read()
entries = re.findall(r'<entry>(.*?)</entry>', data, re.S)
for e in entries:
    title = re.search(r'<title>(.*?)</title>', e, re.S)
    pub = re.search(r'<published>(.*?)</published>', e, re.S)
    idm = re.search(r'<id>(.*?)</id>', e, re.S)
    t = ' '.join(title.group(1).split()) if title else '?'
    p = pub.group(1)[:10] if pub else '?'
    i = idm.group(1) if idm else '?'
    print(f'  [{p}] {t} | {i}')
" >> "$OUT"
  sleep 1
done
echo "Saved to $OUT"
