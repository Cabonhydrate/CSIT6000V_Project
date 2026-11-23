# Frontend Troubleshooting

## PDF.js: `Object.defineProperty called on non-object`

The Next.js 15 Webpack development build can fail when loading certain
versions of `pdfjs-dist` through `react-pdf`. Importing the component with
`ssr: false` alone does not fix a Webpack `eval` sourcemap compatibility bug.

This repository pins `react-pdf` to **10.1.0** (no caret), a version reported
to avoid this bug. Its `pdfjs-dist` worker should be left at the version
resolved by React-PDF; don't mix PDF.js and worker versions.

For an existing checkout, in `frontend` run:

```powershell
npm install react-pdf@10.1.0 --save-exact
Remove-Item -Recurse -Force .next -ErrorAction SilentlyContinue
npm run dev
```

If you still see the same error, try Next.js Turbopack for **development**:

```powershell
npx next dev --turbopack
```

Do not change the `next.config.mjs` Webpack devtool to an unsupported value;
Next.js intentionally manages its devtool in development mode.

The above steps have not been run in a Windows browser in this delivery
runtime. Validate with a real uploaded PDF after installing dependencies.

Relevant upstream reports:
- https://github.com/wojtekmaj/react-pdf/issues/2031
- https://github.com/vercel/next.js/issues/89177
- https://github.com/mozilla/pdf.js/issues/20435
