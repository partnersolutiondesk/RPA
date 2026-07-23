import json


def detect_signature(input_json_string):
    """
    Input:
        {
            "filePath": "<path_to_document_extraction_json>",
            "documentData": {...}
        }

    Returns:
        Updated documentData JSON object.
    """

    payload = json.loads(input_json_string)

    file_path = payload["filePath"]
    document_data = payload["documentData"]

    # Read Document Extraction JSON
    with open(file_path, "r", encoding="utf-8") as f:
        extraction_data = json.load(f)

    feature_objects = (
        extraction_data.get("engineData", {})
        .get("docDetectResult", {})
        .get("featureObjects", [])
    )

    signature_detected = any(
        feature.get("blockType") == "SIGNATURE"
        for feature in feature_objects
    )

    # Ensure path exists
    document_data.setdefault("fields", {})
    document_data["fields"].setdefault("signature", {})

    # Update value
    document_data["fields"]["signature"]["value"] = signature_detected

    # Return only documentData
    return document_data