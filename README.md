# Apache SINGA docs-site (README still work in progress)

## Docusaurus Website Local Set Up

This website was created with [Docusaurus](https://docusaurus.io/).

You need at least `node` and `yarn` to get started with setting up a local development environment.

1. Start from the SINGA root directory, install any website specific dependencies by `yarn install`.

```sh
# Install dependencies
$ yarn install
```

2.  Run a development server with hot-reloading to check changes by running `yarn start` in the website directory.

```sh
# Start the site
$ yarn run start:website
```

## Docs for the Docusaurus Site

All the docs are located in the `docs` folder.

"Versioned documents are placed into `website/versioned_docs/version-${version}`, where `${version}` is the version number you supplied the `version` script... If you wish to change the documentation for a past version, you can access the files for that respective version. (https://docusaurus.io/docs/en/versioning.html#storing-files-for-each-version)

"Documents in the `docs` directory will be considered part of version `next` and they are available, for example, at the URL `docs/next/doc1.html`. Documents from the latest version use the URL `docs/doc1.html`" (https://docusaurus.io/docs/en/versioning)

## News for the Docusaurus Site

All the news are located in the `website/blog/` folder.

# Editing Content

## Static assets

Static assets used in documents and blog/news should go into `docs/assets` and `website/blog/assets`

## Editing an existing docs page

Edit docs by navigating to `docs/` and editing the corresponding document:

`docs/doc-to-be-edited.md`

```markdown
---
id: page-needs-edit
title: This Doc Needs To Be Edited
---

Edit me...
```

For more information about docs, click [here](https://docusaurus.io/docs/en/navigation)

## Editing an existing news post

Edit blog posts by navigating to `website/blog` and editing the corresponding post:

`website/blog/post-to-be-edited.md`

```markdown
---
id: post-needs-edit
title: This Blog Post Needs To Be Edited
---

Edit me...
```

For more information about blog posts, click [here](https://docusaurus.io/docs/en/adding-blog)

# Adding Content

## Adding a new docs page to an existing sidebar

1. Create the doc as a new markdown file in `/docs`, example `docs/newly-created-doc.md`:

```md
---
id: newly-created-doc
title: This Doc Needs To Be Edited
---

My new content here..
```

1. Refer to that doc's ID in an existing sidebar in `website/sidebar.json`:

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

For more information about adding new docs, click [here](https://docusaurus.io/docs/en/navigation)

## Adding a new blog/news post

1. Make sure there is a header link to your blog in `website/siteConfig.js`:

`website/siteConfig.js`

```javascript
headerLinks: [
    ...
    { blog: true, label: 'Blog' },
    ...
]
```

2. Create the blog post with the format `YYYY-MM-DD-My-Blog-Post-Title.md` in `website/blog`:

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

For more information about blog posts, click [here](https://docusaurus.io/docs/en/adding-blog)

## Adding items to your site's top navigation bar

1. Add links to docs, custom pages or external links by editing the headerLinks field of `website/siteConfig.js`:

`website/siteConfig.js`

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

For more information about the navigation bar, click [here](https://docusaurus.io/docs/en/navigation)

## Adding custom pages

1. Docusaurus uses React components to build pages. The components are saved as .js files in `website/pages/en`:
1. If you want your page to show up in your navigation header, you will need to update `website/siteConfig.js` to add to the `headerLinks` element:

`website/siteConfig.js`

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

For more information about custom pages, click [here](https://docusaurus.io/docs/en/custom-pages).

## Deploy

"To create a static build of your website, run the following script from the `website` directory: `yarn run build # or npm run build`"

`cd` to `/website/build/singa/` to serve the static files
