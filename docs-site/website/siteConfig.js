/*
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
*/

/*
 Configuration file: The website/siteConfig.js file 
 is the main configuration file used by Docusaurus.
 See https://docusaurus.io/docs/site-config for all the possible
 site configuration options.
*/

// List of projects/orgs using your project for the users page.
// this field is used by example the pages/en/index.js
// and pages/en/users.js files provided
const users = require("./data/users")

// new github repo for this website Jan2020
const docsRepoUrl = "https://github.com/apache/singa-doc"
const mainRepoUrl = "https://github.com/apache/singa"

const siteConfig = {
  title: "Apache SINGA", // Title for your website.
  tagline: "Distributed deep learning system",
  // temp staging github page using feynmanDNA's github
  // url: "https://feynmandna.github.io",
  url: "https://singa.apache.org", // Your website URL
  baseUrl: "/", // Base URL for your project */
  // For github.io type URLs, you would set the url and baseUrl like:
  //   url: 'https://facebook.github.io',
  //   baseUrl: '/test-site/',

  // display an edit button for docs markdowns
  // docs path after merging with main repo
  editUrl: `${docsRepoUrl}/blob/master/docs-site/docs/`,

  // Used for publishing and more
  projectName: "singa-doc", // cd to /website/build/singa-doc/ to serve
  organizationName: "apache",
  // For top-level user or org sites, the organization is still the same.
  // e.g., for the https://JoelMarcey.github.io site, it would be set like...
  //   organizationName: 'JoelMarcey'

  // For no header links in the top nav bar -> headerLinks: [],
  headerLinks: [
    { doc: "installation", label: "Docs" },
    { doc: "source-repository", label: "Community" },
    // {page: 'help', label: 'Help'},
    { blog: true, label: "News" },
    { href: "https://apache-singa.readthedocs.io/en/latest/", label: "API" },
    // algolia DocSearch
    { search: true },
    // Determines language drop down position among links
    // will need crowdin setup later
    { languages: true },
    // main singa code base on github
    { href: mainRepoUrl, label: "GitHub" },
  ],

  // If you have users set above, you add it here:
  users,

  // disable showing the title
  // in the header next to the header icon
  disableHeaderTitle: true,

  /* path to images for header/footer */
  headerIcon: "img/singa.png",
  footerIcon: "img/singa-logo-square.png",
  favicon: "img/favicon.ico",

  /* algolia DocSearch setup for header */
  algolia: {
    apiKey: "45202133606c0b5fa6d21cddc4725dd8",
    indexName: "apache_singa",
    algoliaOptions: {
      // different search results for the different versions or languages
      // algoliaOptions: { 'facetFilters': ["version:$VERSION", "language:$LANGUAGE"] },
      // So as of Apr 2020 Versions crawled by algolia: 2.0.0, 3.0.0, next
      facetFilters: ["language:en", "version:3.0.0"],
    },
  },

  /* Colors for website */
  colors: {
    // many of the colors are over-written in custom.css
    primaryColor: "#d27400",
    secondaryColor: "#808080", // 2nd layer of toolbar in smaller screen
    $myColor: "#2f3f4d",
  },

  /* Blog setting */
  blogSidebarCount: "ALL", // int N or string "ALL"
  blogSidebarTitle: { default: "Recent News", all: "All News" },

  /* Twitter share at bottom of Blog/News */
  twitter: true,

  /* Custom fonts for website */

  // fonts: {
  //   myFont: [
  //     "Serif",
  //     "Times New Roman",
  //   ],
  //   myOtherFont: [
  //     "-apple-system",
  //     "system-ui"
  //   ]
  // },

  // This copyright info is used in /core/Footer.js and blog RSS/Atom feeds.
  copyright: `Copyright © ${new Date().getFullYear()}
   The Apache Software Foundation. All rights reserved.
   Apache SINGA, Apache, the Apache feather logo, and
   the Apache SINGA project logos are trademarks of The
   Apache Software Foundation. All other marks mentioned
   may be trademarks or registered trademarks of their
   respective owners.`,

  highlight: {
    // Highlight.js theme to use for syntax highlighting in code blocks.
    theme: "atom-one-dark",
  },

  // Add custom scripts here that would be placed in <script> tags.
  scripts: ["https://buttons.github.io/buttons.js"],

  // Adding Google fonts:
  stylesheets: [
    "https://fonts.googleapis.com/css?family=Source+Sans+Pro:400,400i,700",
    "https://fonts.googleapis.com/css2?family=Baloo+Paaji+2&family=Source+Sans+Pro:wght@200;300&display=swap",
  ],

  // On page navigation for the current documentation page.
  onPageNav: "separate",
  // No .html extensions for paths.
  cleanUrl: true,

  // Open Graph and Twitter card images.
  ogImage: "img/singa_twitter_banner.jpeg",
  twitterUsername: "ApacheSINGA",
  twitterImage: "img/singa_twitter_banner.jpeg",

  // For sites with a sizable amount of content, set collapsible to true.
  // Expand/collapse the links and subcategories under categories.
  // docsSideNavCollapsible: true,

  // Show documentation's last contributor's name.
  // enableUpdateBy: true,

  // Show documentation's last update time.
  enableUpdateTime: true,

  // pass down the docsRepoUrl to footer (edit users with PR) etc
  docsRepoUrl,

  scrollToTop: true,
}

module.exports = siteConfig
