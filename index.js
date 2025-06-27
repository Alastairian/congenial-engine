const express = require("express");
const app = express();

app.get("/", (req, res) => {
  res.send("Hello from Vercel and congenial-engine!");
});

module.exports = app; // Important for Vercel's Node.js runtime