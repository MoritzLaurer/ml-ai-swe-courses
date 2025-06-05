# TypeScript Naming Conventions Summary

| Convention             | Case Example     | Typical Usage                                                     |
| :--------------------- | :--------------- | :---------------------------------------------------------------- |
| **`PascalCase`** | `UserProfile`    | Type Aliases, Interfaces, Classes, Enum Names, Enum Members (often) |
| **`camelCase`** | `userProfile`    | Variables, Objects, Function/Method Names               |
| **`snake_case`** | `user_profile`   | **Not Standard** (Except sometimes for external data/config keys) |
| **`SCREAMING_SNAKE_CASE`** | `USER_PROFILE` | True Constants (often primitives), Enum Members (sometimes)         |

**Note:** Consistency within a project is the most important principle. Linters like ESLint are often used to enforce these conventions.