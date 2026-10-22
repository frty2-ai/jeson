# Supported JSON Schema

jeson reads a practical subset of JSON Schema (draft 2020-12 keywords). Run `jeson schema your.schema.json` to see
what each field compiles to.

## Answered today

```json
{
  "type": "object",
  "properties": {
    "signed":       {"type": "boolean", "description": "The signatory line carries a handwritten signature."},
    "currency":     {"type": "string", "enum": ["INR", "USD", "EUR"], "description": "The currency the amounts are in."},
    "risk":         {"type": "string", "enum": ["low", "medium", "high"], "x-ordered": true},
    "line_items":   {"type": "integer", "minimum": 1, "maximum": 10, "description": "How many line items the table lists."},
    "stamp":        {"type": ["boolean", "null"], "description": "A round stamp is on the page; null if the page is cut off."},
    "seller":       {"type": "object", "properties": {
                       "gstin_shown": {"type": "boolean"}}}
  }
}
```

| Keyword | Meaning in jeson |
|---|---|
| `"type": "boolean"` | a boolean field |
| `"enum": [...]` | a choice among the listed values (strings or numbers) |
| `"x-ordered": true` (with `enum`) | the values are ordered levels (an ordinal) |
| `"type": "integer"` with `minimum` and `maximum` (at most 21 values) | an ordinal over the range |
| `"type": [..., "null"]`, or `anyOf` containing `{"type": "null"}` | null is one of the allowed answers |
| `"type": "object"` with `properties` | nested fields, to any depth |
| `description`, `title` | the words the model reads for the field: write them like an instruction to a careful person |

## Recognised, not answered yet

These compile, and are listed under `"_unanswered"` in the output instead of being invented:

| Schema | Kind | Status |
|---|---|---|
| `"type": "string", "x-primitive": "span"` | span: text copied from the page | experimental, see [roadmap](roadmap.md) |
| `"type": "number"`, or `integer` without a small range | number | planned: read as a span, parsed in code |
| array of 4 numbers named `*_box` | box | planned |
| array of `enum` items | multi-label | planned |

## Never answered

Free text (`"type": "string"` without `enum` or `x-primitive`) and arrays of objects need generation. They are
listed under `"_unanswered"`. Repeated structures can be written as fixed sections today (`line_01`, `line_02`, ...
with a `present` boolean each); variable-length arrays are on the roadmap.

## Not supported

`$ref` / `$defs` / `definitions` raise `NotImplementedError` (inline the definitions). `oneOf` / `anyOf` are read
only for their types, enums and null. `pattern`, `format` and numeric bounds other than an ordinal's range are
ignored.

## Forcing a kind

`"x-primitive": "<kind>"` overrides the mapping: one of `noul`, `choice`, `score`, `number`, `span`, `box`,
`multi_label`, `multi_span`, or `generate` (skip the field).

## Writing good descriptions

The field name and description are everything the model knows about a field. What helps, in our runs:

- Say what is visible, not what it means: "A red PAID stamp is on the page", not "The invoice is paid".
- Name the region when the page has several candidates: "The date at the top of the form (not the date of birth)".
- For an absent value, say what to answer: "null if the box is empty".
- For a row in a repeated section, say which row and what to answer when it does not exist.
