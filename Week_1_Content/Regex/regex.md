# Introduction to Regular Expressions

Regular expressions are a powerful tool for pattern matching in text. They are widely used in Natural Language Processing (NLP) for tasks like data cleaning, feature extraction, and information retrieval.

### [Quick Reference Cheat Sheet](https://www.geeksforgeeks.org/python/python-regex-cheat-sheet/)

## Regex101 & Python `re` Module – Key Points

### 1. Regex Basics & Special Sequences
- `.` – matches any character except newline  
- `\d` – digit; `\D` – non-digit  
- `\w` – word char (alphanumeric + `_`); `\W` – non-word  
- `\s` – whitespace; `\S` – non-whitespace  
- Anchors: `^` (start), `$` (end), `\b` (word boundary), `\B` (not boundary)

### 2. Character Classes & Quantifiers
- `[abc]` – any of `a`, `b`, `c`; `[^abc]` – any except  
- `a|b` – match either a or b  
- Quantifiers:
  - `*` – 0 or more  
  - `+` – 1 or more  
  - `?` – 0 or 1  
  - `{n}` – exactly n; `{n,m}` – range of n to m  

### 3. Escaping & Raw Strings
- Escape metacharacters: `.` → `\.`, `*` → `\*`
- Use raw strings: `r"\d+"` to avoid Python escaping

### 4. Common Patterns
- Email: `[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9-.]+`  
- Phone: `\d{3}[-.]\d{3}[-.]\d{4}`  
- Domain/URL: use escaped periods and word boundaries  

### 5. Flags
- `re.IGNORECASE` for case-insensitive matching

### 6. Useful Methods
- `re.compile(pattern)` – compile regex  
- `finditer()` – returns iterator of match objects with span/index  
- `match()`, `search()` for different matching behaviors  

### 7. Testing Tools
- Recommend using **[Regex101.com](https://regex101.com)** to experiment and visualize patterns

## References Used
1. Video: [Python Tutorial: re Module - How to Write and Match Regular Expressions (Regex)](https://www.youtube.com/watch?v=K8L6KVGG-7o)
2. Article: [Regex101 Documentation GFG](https://www.geeksforgeeks.org/python/python-regex/)
