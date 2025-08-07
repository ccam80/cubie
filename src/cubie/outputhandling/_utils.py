# def parse_string_with_value(text: str) -> Tuple[str, Optional[int]]:
#     """For a string of the form "text[int]", return ("text", int) or (
#     "text",None) for a string without square
#     brackets."""
#     m = re.search(r'\[(\d+)]', text)
#     number = int(m.group(1)) if m else None
#     parsed_text = re.sub(r'\[\d+]', '', text).strip()
#     return parsed_text, number
#
# def process_outputs_list(outputs: list[str]) -> Tuple[list[str], dict[str, int]]:
#     """Process a list of outputs to separate names and values."""
#     names = []
#     ns = {}
#     for output in outputs:
#         (type_string, n) = parse_string_with_value(output)
#         names.append(type_string)
#         if n is not None:
#             ns.update({f"n_{type_string}": n})
#
#     return names, ns
