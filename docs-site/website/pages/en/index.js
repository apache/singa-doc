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
const MarkdownBlock = CompLibrary.MarkdownBlock
const Showcase = require(`${process.cwd()}/core/Showcase.js`)
// TODO: add <translate> tags
// const translate = require('../../server/translate.js').translate;

//
const iconGr = require("react-icons/gr")
const iconGi = require("react-icons/gi")
const iconBs = require("react-icons/bs")
const iconFa = require("react-icons/fa")
const iconCg = require("react-icons/cg")
const iconIm = require("react-icons/im")

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

        <div className="mainPageContainer">
          <Container padding="top, bottom">
            <div
              style={{
                fontWeight: "bold",
                textAlign: "left",
                fontSize: "35px",
                paddingBottom: "10px",
              }}
            >
              Features
            </div>
            <div className="container-2">
              <div className="container-2-box">
                <iconGr.GrInstall className="icon" />
                <h2>Easy installation</h2>
                <p>
                  <MarkdownBlock>
                    Easy installation using
                    [Conda](https://singa.apache.org/docs/installation/#using-conda),
                    [Pip](https://singa.apache.org/docs/installation/#using-pip),
                    [Docker](https://singa.apache.org/docs/installation/#using-docker)
                    and [from
                    Source](https://singa.apache.org/docs/installation/#from-source)
                  </MarkdownBlock>
                </p>
              </div>
              <div className="container-2-box">
                <iconGi.GiMeepleCircle className="icon" />
                <h2>Model zoo</h2>
                <p>
                  <MarkdownBlock>
                    Various example deep learning models are provided in SINGA
                    repo on
                    [Github](https://github.com/apache/singa/tree/master/examples)
                    and on [Google Colab](https://colab.research.google.com/)
                  </MarkdownBlock>
                </p>
              </div>
              <div className="container-2-box">
                <iconBs.BsDistributeVertical className="icon" /> 
                <h2>Distributed training</h2>
                <p>
                  <MarkdownBlock>
                    SINGA supports data parallel training across multiple GPUs
                    (on a single node or across different nodes)
                  </MarkdownBlock>
                </p>
              </div>
              <div className="container-2-box">
                <iconFa.FaCalculator className="icon" /> 
                <h2>Automatic gradient calculation</h2>
                <p>
                  <MarkdownBlock>
                    SINGA records the [computation
                    graph](https://singa.apache.org/docs/graph/) and applies the
                    backward propagation automatically after forward propagation
                  </MarkdownBlock>
                </p>
              </div>
              <div className="container-2-box">
                <iconFa.FaMemory className="icon" style={{fontSize: "22px"}}/>
                <h2>Memory optimization</h2>
                <p>
                  <MarkdownBlock>
                    The optimization of memory are implemented in the
                    [Device](https://singa.apache.org/docs/device/) class
                  </MarkdownBlock>
                </p>
              </div>
              <div className="container-2-box">
                <iconGr.GrOptimize className="icon" /> 
                <h2>Parameter optimization</h2>
                <p>
                  <MarkdownBlock>
                    SINGA supports various popular optimizers including
                    stochastic gradient descent with momentum, Adam, RMSProp,
                    and AdaGrad, etc
                  </MarkdownBlock>
                </p>
              </div>
              <div className="container-2-box">
                <iconCg.CgArrowsExchangeAlt className="icon" style={{fontSize: "30px"}}/>
                <h2>Interoperability</h2>
                <p>
                  <MarkdownBlock>
                    SINGA supports loading [ONNX](https://onnx.ai/) format
                    models and saving models defined using SINGA APIs into ONNX
                    format, which enables AI developers to use models across
                    different libraries and tools
                  </MarkdownBlock>
                </p>
              </div>
              <div className="container-2-box">
                <iconIm.ImClock className="icon" />
                <h2>Time profiling</h2>
                <p>
                  <MarkdownBlock>
                    SINGA supports the time profiling of each of the operators
                    buffered in the graph
                  </MarkdownBlock>
                </p>
              </div>
              <div className="container-2-box">
                <iconBs.BsHeptagonHalf className="icon" /> 
                <h2>Half precision</h2>
                <p>
                  <MarkdownBlock>
                    Half precision is supported to bring benefits, e.g., less
                    GPU memory, supporting larger networks and faster training,
                    etc
                  </MarkdownBlock>
                </p>
              </div>
            </div>
          </Container>
          <Container padding="top, bottom" background="light">
            <p
              style={{
                textAlign: "center",
                fontSize: "24px",
                fontWeight: "bold",
              }}
            >
              SINGA has a well architected software stack and easy-to-use Python interface to improve
              usability
            </p>
            <img
              style={{ width: "35%" }}
              className="containerImg"
              src={`${siteConfig.baseUrl}img/singav3-sw.png`}
              alt="Usability"
            />
          </Container>
          <Container padding="top, bottom">
            <p
              style={{
                textAlign: "center",
                fontSize: "24px",
                fontWeight: "bold",
              }}
            >
              SINGA parallelizes the training and optimizes the communication
              cost to improve training scalability
            </p>
            <img
              style={{ width: "27%" }}
              className="containerImg"
              src={`${siteConfig.baseUrl}img/benchmark.png`}
              alt="Scalability"
            />
          </Container>
          <Container padding="top, bottom" background="light">
            <p
              style={{
                textAlign: "center",
                fontSize: "24px",
                fontWeight: "bold",
              }}
            >
              SINGA builds a computational graph to optimize the training speed
              and memory footprint
            </p>
            <img
              style={{ width: "37%" }}
              className="containerImg"
              src={`${siteConfig.baseUrl}img/GraphOfMLP.png`}
              alt="Efficiency"
            />
          </Container>

          <Container padding={["bottom", "top"]}>
            <div
              style={{
                fontWeight: "bold",
                textAlign: "left",
                fontSize: "30px",
                paddingBottom: "40px",
              }}
            >
              Install SINGA
            </div>

            <div className="container-3">
              <button className="container-3-button">
                <a href="https://singa.apache.org/docs/installation/#using-pip">
                  Using Pip
                  <span style={{ float: "right" }}>
                    <iconBs.BsDownload />
                  </span>
                </a>
              </button>
              <button className="container-3-button">
                <a href="https://singa.apache.org/docs/installation/#using-docker">
                  Using Docker
                  <span style={{ float: "right" }}>
                    <iconBs.BsDownload />
                  </span>
                </a>
              </button>
              <button className="container-3-button">
                <a href="https://singa.apache.org/docs/installation/#from-source">
                  From Source
                  <span style={{ float: "right" }}>
                    <iconBs.BsDownload />
                  </span>
                </a>
              </button>
            </div>
          </Container>
          <Container>
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
          </Container>
        </div>
      </div>
    )
  }
}

module.exports = Index
