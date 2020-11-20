# Apache SINGA Website

## Local Set Up

This website is created with [Docusaurus](https://docusaurus.io/). To set up the
website locally,

1. Install [yarn](https://classic.yarnpkg.com/en/docs/install/#debian-stable)

2. Install [node](https://askubuntu.com/a/548776) (version>=10).

3. Install dependencies

```sh
# in singa-doc/docs-site folder
$ yarn install
```

4. Run a development server with hot-reloading which updates the webpages
   immediately upon the modifications to the source files,

```sh
# in singa-doc/docs-site folder
$ yarn run start:website
```

## Generate Static HTML Files for Deployment

To create a static build of your website, run the following script from the
`website` directory:

```sh
$ yarn run build # or npm run build
```

The generated html files are under `website/build/singa-doc/`.

To host the website locally for testing, under `website/build/singa-doc/` type:

```sh
$ python -m http.server
```

## Update and Add Pages

### Navigation Bar

To add links to docs, custom pages or external website to the top navigation
bar, edit the headerLinks field of `website/siteConfig.js`:

```javascript
{
  headerLinks: [
    ...
    /* you can add docs */
    { doc: 'my-examples', label: 'Examples' },
    /* you can add custom pages */
    { page: 'help', label: 'Help' },
    /* you can add external links */
    { href: 'https://github.com/facebook/docusaurus', label: 'GitHub' },
    ...
  ],
  ...
}
```

For more information about the navigation bar, click
[here](https://docusaurus.io/docs/en/navigation)

### Documentation

All the technical documents are located in the `singa-doc/doc-site/docs` folder.
They are considered as the documentation for `next`
[version](https://docusaurus.io/docs/en/versioning). The URLs for these webpages
are like `docs/next/xxx.html`.

[Versioned documents](https://docusaurus.io/docs/en/versioning.html#storing-files-for-each-version)
are in `website/versioned_docs/version-${version}`, where `${version}` is the
version number.  
The URLs for the webpages of a specific version are
`docs/version-${version}/xxx.html`. For the latest version, you can also visit
the webpages using `docs/xxx.html` (without specifying the version). Suppose the
current latest version is v2.0 and we are going to release v3.0. By running

```sh
yarn run version 3.0.0
```

The documents for the current _next_ version will be copied into
`website/versioned_docs/version-3.0.0`.

To add a new document for the next version,

1. Create the doc as a new markdown file in `/docs`, example
   `docs/newly-created-doc.md`:

```md
---
id: newly-created-doc
title: This Doc Needs To Be Edited
---

My new content here..
```

2. Refer to that doc's ID in an existing sidebar in `website/sidebar.json`:

```javascript
// Add newly-created-doc to the Getting Started category of docs
{
  "docs": {
    "Getting Started": [
      "quick-start",
      "newly-created-doc" // new doc here
    ],
    ...
  },
  ...
}
```

If the sidebar does not exist, you need to add it in the `sidebar.json` file.

Static assets are under `docs/assets`. For more information about adding new
docs, click [here](https://docusaurus.io/docs/en/navigation)

### News

News posts are added as blog posts. To add a news post, create a new file with
the format `YYYY-MM-DD-My-Blog-Post-Title.md` in
`singa-doc/doc-site/website/blog`, e.g.,
`website/blog/2018-05-21-New-Blog-Post.md`

```markdown
---
author: Frank Li
authorURL: https://twitter.com/foobarbaz
authorFBID: 503283835
title: New Blog Post
---

Lorem Ipsum...
```

There is a link from the top navigation bar to the blog view due to the
following setting in `website/siteConfig.js`:

```javascript
headerLinks: [
    ...
    { blog: true, label: 'Blog' },
    ...
]
```

Static assets are under `website/blog/assets`. For more information about blog
posts, click [here](https://docusaurus.io/docs/en/adding-blog).

### Customized Pages

Docusaurus uses React components to build pages. The components are saved as .js
files in `website/pages/en`: If you want your page to show up in your navigation
header, you will need to update `website/siteConfig.js` to add to the
`headerLinks` element:

```javascript
{
  headerLinks: [
    ...
    { page: 'my-new-custom-page', label: 'My New Custom Page' },
    ...
  ],
  ...
}
```

For more information about custom pages, click
[here](https://docusaurus.io/docs/en/custom-pages).
