# Python File Opening Modes: `a`, `a+`, `w`, `w+`, `r+`

## Overview

Python’s built-in `open()` function allows files to be opened in different modes depending on whether we want to **read**, **write**, or **append** data.  
Understanding these modes is critical to avoid **data loss**, **unexpected overwrites**, and **file pointer errors**.

This document explains the differences between the following modes:

- `a` (append only)
- `a+` (append + read)
- `w` (write only)
- `w+` (write + read)
- `r+` (read + write)

---

## Key Concept: The `+` Symbol

The `+` symbol **always enables both reading and writing**.  
However, **it does NOT change where the file pointer starts** — that depends on the base mode (`r`, `w`, or `a`).

This is where many mistakes happen.

---

## Summary Table

| Mode | Read | Write            | File Exists | File Does Not Exist | Pointer Position | Truncates File |
| ---- | ---- | ---------------- | ----------- | ------------------- | ---------------- | -------------- |
| `a`  | ❌   | ✅ (append only) | Appends     | Creates file        | End              | ❌             |
| `a+` | ✅   | ✅ (append only) | Appends     | Creates file        | End              | ❌             |
| `w`  | ❌   | ✅               | Overwrites  | Creates file        | Start            | ✅             |
| `w+` | ✅   | ✅               | Overwrites  | Creates file        | Start            | ✅             |
| `r+` | ✅   | ✅               | Modifies    | ❌ Error            | Start            | ❌             |

---

## Mode-by-Mode Explanation

### 1. Append Mode (`a`)

- Writes data **only at the end**
- Existing content is preserved
- File is created if it does not exist
- Reading is not allowed

```python
with open("example.txt", "a") as file:
    file.write("Appended text\n")
```
