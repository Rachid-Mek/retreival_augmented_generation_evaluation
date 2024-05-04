def extract_text(results):
  """
  Extracts informative text from machine responses, handling specific formatting.

  Args:
      results: A list of strings containing machine responses.

  Returns:
      A list of strings containing the extracted text from each response.
  """

  extracted_texts = []

  for text in results:
    # Split text based on special markers
    parts = text.split("<|eot_id|>")  # Split at end-of-turn marker
    if len(parts) > 1:
      # Extract text after assistant header
      assistant_response = parts[-1].split("<|start_header_id|>assistant<|end_header_id|>")
      if len(assistant_response) > 1:
        extracted_texts.append(assistant_response[1].strip())  # Extract and clean assistant response
      else:
        print(f"Assistant response not found for: {text}")  # Informative message for missing response
    else:
      print(f"Unexpected format for response: {text}")  # Informative message for unexpected format

  return extracted_texts