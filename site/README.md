# site/ — GitHub Pages source

Static site for <https://decepticons.win>. No build step, no Jekyll. Plain HTML + CSS.

## Preview locally

```bash
python3 -m http.server -d site 8000
# open http://localhost:8000
```

## Deployment

Auto-deployed by `.github/workflows/pages.yml` on every push to `main` that
touches `site/**`. The workflow uploads this directory as a Pages artifact and
publishes via `actions/deploy-pages`.

Required GitHub setting: **Settings → Pages → Source = GitHub Actions**.

## Custom domain

`CNAME` contains the apex domain (`decepticons.win`). GitHub Pages reads it
from the deployed artifact root. DNS for the domain must point at GitHub Pages
(four `A` records to GitHub's apex IPs, or one `ALIAS`/`ANAME` if your registrar
supports it; plus a `CNAME` for `www` if used).

## Files

- `index.html` — landing page
- `styles.css` — single stylesheet, dark theme
- `assets/logo.webp` — project logo
- `CNAME` — custom domain
- `.nojekyll` — disables Jekyll processing on Pages
