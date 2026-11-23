const CONTRAST = /\b(?:but|although|though|however|yet|even so|still)\b/i;
const NEGATION = /\b(?:(?:not|never|hardly|no longer)|[a-z]+n't)(?:\s+(?:\w+|[a-z]+'[a-z]+)){0,3}\s*$/i;
const alternatives = patterns => new RegExp(patterns.map(pattern => pattern.source).join("|"), "gi");

const RELIEF_AT_END = alternatives([
  /\b(?:relieved|glad|thankful)\s+to\s+(?:finally\s+)?(?:leave|get out|check out|be leaving|be done)\b/,
  /\b(?:relieved|glad|thankful)\s+(?:when|once|that)\s+(?:it|this)\s+(?:was\s+(?:finally\s+)?(?:over|time to leave)|(?:finally\s+)?ended)\b/,
  /\b(?:relieved|glad|thankful)\s+(?:when|once|that)\s+(?:i|we)\s+(?:(?:could|got to)\s+(?:finally\s+)?leave|(?:finally\s+)?(?:left|got out))\b/,
]);
const PRAISE_LEAD = /^\s*["']?(?:well,?\s+)?(?:perfect(?: timing)?|great(?: timing| job| news)?|wonderful|brilliant|awesome|fantastic|excellent|amazing|just what i needed|exactly what i needed|thanks a lot)["']?\s*[,!.:;—–-]/i;
const PRAISE_PREFIX = /^\s*(?:i\s+)?love\s+(?:that|how)\s+/i;
const software = /\b(?:app|application|browser|computer|editor|form|game|laptop|page|portal|program|screen|software|system|website)\s+/;
const failure = /(?:(?:just|kept|still|suddenly|completely)\s+){0,2}(?:froze|freez(?:e[sd]?|ing)|crash(?:ed|es|ing)?|stopped responding|locked up|hung|reset(?:s|ting)?|wouldn't (?:load|open|save|submit))\b/;
const TECH_FAILURE = new RegExp(software.source + failure.source, "gi");
const LOST_WORK = /\b(?:lost|deleted|erased|wiped|discarded)\s+(?:all\s+(?:of\s+)?)?(?:my|our|the)\s+(?:work|files?|notes|progress|drafts?|entries|changes|answers|data)\b/gi;
const REPEATED_WORK = alternatives([
  /\b(?:retyp(?:e|ed|ing)|reenter(?:ed|ing)?|start(?:ed|ing)? over)\b/,
  /\b(?:typ(?:e|ed|ing)|enter(?:ed|ing)?|fill(?:ed|ing)? (?:in|out)|submit(?:ted|ting)?|upload(?:ed|ing)?)\b[^.!?;\n]{0,36}\b(?:again|for (?:a|the) (?:second|third|fourth|fifth) time)\b/,
]);
const INCONVENIENCE = alternatives([
  /\banother\s+(?:(?:\d+|one|two|three|four|five|ten|twenty|thirty|forty|sixty)\s+)?(?:minutes?|hours?|days?)\s+(?:waiting|on hold|in line|of waiting)\b/,
  /\b(?:missed|missing)\s+(?:our|my|the)\s+(?:flight|train|appointment)\b/,
  /\b(?:had to|still need to)\s+pay\s+for\s+(?:it|a replacement)\s+again\b/,
  /\banother\s+(?:(?:app|browser|software)\s+)?(?:crash|freeze|reset)\b/,
]);
const RECOVERY = alternatives([
  /\b(?:recovered|restored|saved)\s+(?:everything|(?:all\s+(?:of\s+)?)?(?:my|our|the)\s+(?:work|files|notes|progress|changes|data))\b/,
  /\b(?:my|our|the)\s+(?:work|files|notes|progress|changes|data)\s+(?:was|were|is|are)\s+(?:all\s+)?(?:safe|intact|recovered|restored)\b/,
  /\b(?:fixed|resolved)\s+(?:everything|it|the (?:issue|problem))\b/,
  /\b(?:was|were|is|are)\s+(?:finally\s+)?(?:fixed|resolved)\b/,
  /\b(?:worked|works|working)\s+(?:well|perfectly|smoothly|again)\b/,
  /\b(?:would|will)\s+(?:happily\s+|definitely\s+)?(?:book|buy|stay|visit|order|use|return|come back)\b[^.!?;\n]{0,24}\bagain\b/,
]);

function lastAffirmed(pattern, text) {
  let last = -1;
  for (const match of text.matchAll(pattern)) {
    const before = text.slice(Math.max(0, match.index - 60), match.index);
    const after = text.slice(match.index + match[0].length);
    if (NEGATION.test(before) || /^\s*,?\s*(?:if|unless)\b/i.test(after)) continue;
    last = match.index;
  }
  return last;
}

function reviewNegativeContext(text) {
  const normalized = text.replace(/[’‘]/g, "'").replace(/[“”]/g, '"').trim();
  const relief = lastAffirmed(RELIEF_AT_END, normalized);
  const recovery = lastAffirmed(RECOVERY, normalized);

  // Relief can describe an improvement. Look for a contrast leading into relief about leaving.
  if (relief >= 0 && recovery < relief && CONTRAST.test(normalized.slice(0, relief))) {
    return "Relief about leaving points to dissatisfaction with the experience.";
  }

  const fault = lastAffirmed(TECH_FAILURE, normalized);
  const loss = lastAffirmed(LOST_WORK, normalized);
  const repetition = lastAffirmed(REPEATED_WORK, normalized);
  const inconvenience = lastAffirmed(INCONVENIENCE, normalized);
  const setback = Math.max(fault, loss, inconvenience);
  const laterRecovery = recovery >= 0 && (
    recovery > Math.max(setback, repetition) ||
    // A repair can be mentioned before the fault when linked with "after".
    /^\b[^.!?;\n]*\bafter\b/i.test(normalized.slice(recovery, setback))
  );

  if (!laterRecovery && fault >= 0 && Math.max(loss, repetition) > fault) {
    return "The software failure is followed by lost work or having to repeat a task.";
  }
  if (!laterRecovery && setback >= 0 && (PRAISE_LEAD.test(normalized) || PRAISE_PREFIX.test(normalized))) {
    return "The opening praise is followed by a concrete setback, which suggests sarcasm.";
  }
  return "";
}

export function applyContext(text, result) {
  if (result.error || result.label === "Negative") return result;
  const reason = reviewNegativeContext(text);
  if (!reason) return result;

  return {
    ...result,
    label: "Negative",
    confidence: null,
    context: { reason, modelLabel: result.label },
  };
}
