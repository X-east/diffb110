# Communication & Language Guidelines

## Language Response Rule

**Primary Rule**: Respond in the same language as the user's question/prompt.

- If user asks in **Chinese (中文)** → respond in **Chinese**
- If user asks in **English** → respond in **English**
- If user mixes languages → prioritize the primary language of the question

## Code Language

**All code must remain in English**, regardless of the language used in questions:
- Function names, variable names, comments
- Docstrings and type hints
- Configuration files and error messages
- Log output

## Documentation Language

- **Markdown files (.md)**: Follow the language of the question
- **Python files (.py)**: Always English
- **Configuration files (.yaml, .json)**: Always English

## Consistency

Maintain consistent language throughout a single response:
- Do not switch between Chinese and English mid-response
- If clarification is needed, ask in the same language as the original question
- Keep technical terminology consistent with project conventions

## Examples

| User Input | Response Language | Code Language |
|-----------|------------------|---------------|
| 中文提问 | 中文回答 | English (code only) |
| English question | English response | English (code only) |
| 混合语言提问 | 按主要语言回答 | English (code only) |
