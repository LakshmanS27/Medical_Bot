import os
import xml.etree.ElementTree as ET
import pandas as pd

def parse_medquad_to_csv(data_dir="MedQuAD", output_csv="dataset.csv"):
    qa_pairs = []

    for root_dir, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".xml"):
                xml_path = os.path.join(root_dir, file)
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()

                    qa_section = root.find("QAPairs")
                    if qa_section:
                        for pair in qa_section.findall("QAPair"):
                            question = pair.findtext("Question")
                            answer = pair.findtext("Answer")

                            if question and answer:
                                qa_pairs.append({"prompt": question.strip(), "response": answer.strip()})
                except Exception as e:
                    print(f"❌ Skipped {xml_path} due to error: {e}")

    if qa_pairs:
        df = pd.DataFrame(qa_pairs)
        df.to_csv(output_csv, index=False)
        print(f"✅ Saved {len(df)} QA pairs to {output_csv}")
    else:
        print("⚠️ No QA pairs found.")

if __name__ == "__main__":
    parse_medquad_to_csv()
