"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { marked } = require("marked");

const source = path.join(__dirname, "실험_연대기.md");
const target = path.join(__dirname, "실험_연대기.html");
const markdown = fs.readFileSync(source, "utf8");
const rendered = marked.parse(markdown, { gfm: true, breaks: false });
const document = `<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>실험 연대기</title>
  <style>
    body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; color: #222; }
    h1, h2, h3 { line-height: 1.25; }
    h2 { margin-top: 2.5rem; border-top: 1px solid #ddd; padding-top: 1rem; }
    table { border-collapse: collapse; display: block; overflow-x: auto; max-width: 100%; }
    th, td { border: 1px solid #bbb; padding: .45rem .6rem; text-align: left; white-space: nowrap; }
    th { background: #f2f2f2; }
    code { background: #f4f4f4; padding: .1rem .25rem; overflow-wrap: anywhere; }
    li { margin-bottom: .25rem; }
  </style>
</head>
<body>
${rendered}</body>
</html>
`;
fs.writeFileSync(target, document, "utf8");
console.log(`Rendered ${path.basename(source)} to ${path.basename(target)}`);
