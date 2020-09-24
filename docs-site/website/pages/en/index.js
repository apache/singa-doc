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
 Pages: The website/pages directory
 contains example top-level pages for the site.
 index.js is the landing page
*/

const React = require("react")

const CompLibrary = require("../../core/CompLibrary.js")

const Container = CompLibrary.Container
const GridBlock = CompLibrary.GridBlock
const Showcase = require(`${process.cwd()}/core/Showcase.js`)
// TODO: add <translate> tags
// const translate = require('../../server/translate.js').translate;

const siteConfig = require(`${process.cwd()}/siteConfig.js`)

function docUrl(doc, language) {
  return siteConfig.baseUrl + "docs/" + (language ? language + "/" : "") + doc
}

function pageUrl(page, language) {
  return siteConfig.baseUrl + (language ? language + "/" : "") + page
}

function HomeSplash(props) {
  const { siteConfig, language } = props

  return (
    <div className="index-hero">
      {/* Increase the network loading priority of the background image. */}
      <img
        style={{ display: "none" }}
        src={`${siteConfig.baseUrl}img/sg-botanic-coleen-rivas-unsplash.jpg`}
        alt="increase priority"
      />
      <div className="index-hero-inner">
        <img
          alt="SINGA-hero-banner"
          className="index-hero-logo"
          src={`${siteConfig.baseUrl}img/singa.png`}
        />
        <h1 className="index-hero-project-tagline">
          A Distributed Deep Learning Library
        </h1>
        <div className="index-ctas">
          <a
            className="button index-ctas-get-started-button"
            href={`${docUrl("installation", language)}`}
          >
            Get Started
          </a>
          <span className="index-ctas-github-button">
            <iframe
              src="https://ghbtns.com/github-btn.html?user=apache&amp;repo=singa&amp;type=star&amp;count=true&amp;size=large"
              frameBorder={0}
              scrolling={0}
              width={160}
              height={30}
              title="GitHub Stars"
            />
          </span>
        </div>
      </div>
    </div>
  )
}

class Index extends React.Component {
  render() {
    const { config: siteConfig, language = "en" } = this.props
    const pinnedUsersToShowcase = siteConfig.users.filter(user => user.pinned)

    return (
      <div>
        <HomeSplash siteConfig={siteConfig} language={language} />
        <div className="announcement">
          <div className="announcement-inner">
            Apache SINGA is an{" "}
            <a href="https://blogs.apache.org/foundation/entry/the-apache-software-foundation-announces57">
              Apache Top Level Project
            </a>
            , focusing on distributed training of deep learning and machine
            learning models
          </div>
        </div>
        <div className="mainContainer">
          <Container padding={["bottom", "top"]} className="mainPageContainer">
            <GridBlock
              contents={[
                {
                  content: `SINGA [parallelizes the training and optimizes the communication cost](./docs/dist-train) to improve training scalability.`,
                  imageAlign: "left",
                  image: `${siteConfig.baseUrl}img/benchmark.png`,
                  imageAlt: "Scalbility",
                  title: "Scalablility",
                },
              ]}
              layout="twoColumn"
            />
          </Container>
          <Container
            padding={["bottom", "top"]}
            className="mainPageContainer"
            background="light"
          >
            <GridBlock
              contents={[
                {
                  content: `SINGA [builds a computational graph](./docs/graph) to optimizes the training speed and memory footprint.`,
                  imageAlign: "right",
                  image: `${siteConfig.baseUrl}img/GraphOfMLP.png`,
                  imageAlt: "Efficiency",
                  title: "Efficiency",
                },
              ]}
              layout="twoColumn"
            />
          </Container>
          <Container padding={["bottom", "top"]} className="mainPageContainer">
            <GridBlock
              contents={[
                {
                  content: `SINGA has a simple [software stack and Python interface](./docs/software-stack) to improve usability.`,
                  imageAlign: "left",
                  image: `${siteConfig.baseUrl}img/singav3.1-sw.png`,
                  imageAlt: "Usability",
                  title: "Usability",
                },
              ]}
              layout="twoColumn"
            />
          </Container>
          <div className="productShowcaseSection paddingBottom">
            <h2 style={{ color: "#904600" }}>Users of Apache SINGA</h2>
            <p>
              Apache SINGA powers the following organizations and companies...
            </p>
            <Showcase users={pinnedUsersToShowcase} />
            <div className="more-users">
              <a className="button" href={`${pageUrl("users", language)}`}>
                All Apache SINGA Users
              </a>
            </div>
          </div>
        </div>
      </div>
    )
  }
}

module.exports = Index
