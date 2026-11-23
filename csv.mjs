export function parseCSV(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  let closed = false;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (quoted) {
      if (char === '"' && text[index + 1] === '"') {
        field += '"';
        index += 1;
      } else if (char === '"') {
        quoted = false;
        closed = true;
      } else {
        field += char;
      }
    } else if (char === "," || char === "\n" || char === "\r") {
      row.push(field);
      field = "";
      closed = false;
      if (char !== ",") {
        if (char === "\r" && text[index + 1] === "\n") index += 1;
        if (row.some(cell => cell.trim())) rows.push(row);
        row = [];
      }
    } else if (closed) {
      if (char !== " " && char !== "\t") throw new Error("A CSV field has text after its closing quote.");
    } else if (char === '"' && field === "") {
      quoted = true;
    } else {
      field += char;
    }
  }
  if (quoted) throw new Error("A CSV field has an unclosed quote.");
  row.push(field);
  if (row.some(cell => cell.trim())) rows.push(row);
  return rows;
}

export function csvValue(value) {
  const text = String(value ?? "");
  // Spreadsheet apps can execute certain imported cells as formulas.
  const safe = /^[\t\r\n]|^\s*[=+@-]/.test(text) ? "'" + text : text;
  return '"' + safe.replaceAll('"', '""') + '"';
}
