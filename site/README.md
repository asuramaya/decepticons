# site/ — GitHub Pages draft

Draft static site for the project. No build step, no Jekyll. Plain HTML + CSS.

## Preview locally

```bash
python3 -m http.server -d site 8000
# open http://localhost:8000
```

## Deploy options

### A. Serve from the `gh-pages` branch via Actions

Use the workflow at `.github/workflows/pages.yml` (included). It runs on every push to
`main` that touches `site/**` and publishes the directory.

In the GitHub repo settings, set:

- **Settings → Pages → Build and deployment → Source** to **GitHub Actions**.

### B. Serve from `/docs`

GitHub Pages can serve from a `/docs` folder on `main`, but the existing `/docs` is
full of Markdown documentation. If you want to use that path:

1. Move the contents of `site/` into `docs/` (will collide with markdown files —
   pick one).
2. Keep the `.nojekyll` file so Jekyll does not try to render the `.md` files.
3. **Settings → Pages → Source** → **Deploy from branch** → `main` / `/docs`.

The included workflow (option A) is the cleaner path.

## Files

- `index.html` — landing page
- `styles.css` — single stylesheet, dark theme
- `assets/logo.webp` — copy of the project logo
- `.nojekyll` — disables Jekyll processing on Pages
