import requests
from bs4 import BeautifulSoup
import re
import json
import os
from datetime import datetime


def clean_js_data(js_data):
    js_data = re.sub(r"//.*?\n", "", js_data)
    js_data = re.sub(r"/\*.*?\*/", "", js_data, flags=re.DOTALL)
    js_data = re.sub(r",(\s*[}\]])", r"\1", js_data)
    js_data = re.sub(r"(\w+):", r'"\1":', js_data)
    return js_data


def extract_model_comparison(soup):
    try:
        script = soup.find("script", string=re.compile("const tableData ="))
        if script:
            data_match = re.search(r"const tableData = (\[.*?\]);", script.string, re.DOTALL)
            if data_match:
                js_data = data_match.group(1)
                json_data = clean_js_data(js_data)
                return json.loads(json_data)
    except Exception as e:
        print(f"Failed to extract model comparison data: {str(e)}")
    return None


def extract_table_by_section(soup, section_id, title_text=None):
    try:
        section = soup.find("div", {"id": section_id})
        if not section:
            print(f"Section {section_id} not found")
            return None

        if title_text:
            title = section.find(text=re.compile(title_text, re.IGNORECASE))
            if not title:
                print(f"Title '{title_text}' not found in section {section_id}")
                return None

        table = section.find("table")
        if not table:
            print(f"No table found in section {section_id}")
            return None

        headers = [th.text.strip() for th in table.find_all("th")]
        if not headers:
            headers = [td.text.strip() for td in table.find("tr").find_all("td")]

        rows = []
        for tr in table.find_all("tr")[1:]:
            cells = [td.text.strip() for td in tr.find_all("td")]
            if cells and len(cells) == len(headers):
                row_data = dict(zip(headers, cells))
                rows.append(row_data)
        return rows

    except Exception as e:
        print(f"Failed to extract table from section {section_id}: {str(e)}")
        return None


def extract_dynamic_list_data(soup):
    try:
        headers_div = soup.find("div", class_="quick-stack-headers")
        if not headers_div:
            return None

        headers = []
        for cell in headers_div.find_all("div", class_="w-layout-cell"):
            title = cell.find("div", class_=["table_title", "model_name"])
            if title:
                headers.append(title.text.strip())

        rows = []
        items = soup.find_all("div", role="listitem")
        for item in items:
            cells = item.find_all("div", class_=["w-layout-cell", "model-cell", "cell-6"])
            row_data = {}

            for i, cell in enumerate(cells):
                if i < len(headers):
                    if "model-cell" in cell.get("class", []) or "cell-6" in cell.get("class", []):
                        model_name = cell.find("div", class_="model_name")
                        if model_name:
                            row_data[headers[i]] = model_name.text.strip()
                    else:
                        text_div = cell.find("div", class_=["table-text", "table-tecx"])
                        if text_div:
                            value = text_div.text.strip()
                            if value:
                                row_data[headers[i]] = value

                        cost_divs = cell.find_all("div", class_="table-text")
                        if len(cost_divs) == 2:
                            values = [div.text.strip() for div in cost_divs if div.text.strip()]
                            if values:
                                row_data[headers[i]] = " / ".join(values)

            if row_data:
                rows.append(row_data)

        return rows

    except Exception as e:
        print(f"Failed to extract dynamic list data: {str(e)}")
        return None


def fetch_leaderboard_data(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        data = {}

        model_data = extract_model_comparison(soup)
        if model_data:
            data["model_comparison"] = model_data
            print("Successfully extracted model comparison data")

        cost_data = extract_table_by_section(soup, "cost-context", "Cost and Context Window")
        if cost_data:
            data["cost_context"] = cost_data
            print("Successfully extracted cost context data")

        coding_data = extract_table_by_section(soup, "coding", "HumanEval")
        if coding_data:
            data["humaneval_coding"] = coding_data
            print("Successfully extracted humaneval coding data")

        dynamic_data = extract_dynamic_list_data(soup)
        if dynamic_data:
            data["model_details"] = dynamic_data
            print("Successfully extracted model details data")

        if not data:
            raise ValueError("No table data found")

        return data

    except Exception as e:
        print(f"Failed to fetch data: {str(e)}")
        return None


def save_data(data, base_path):
    save_dir = os.path.join(base_path, "leaderboard")
    os.makedirs(save_dir, exist_ok=True)

    for table_name, table_data in data.items():
        json_path = os.path.join(save_dir, f"llm_vellu_ai_{table_name}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(table_data, f, indent=2, ensure_ascii=False)
        print(f"Saved {table_name} data to: {json_path}")


def analyze_performance_metrics(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        performance_data = []
        for model in data:
            model_name = model.get("Model", "")
            throughput = model.get("Throughput", "")
            latency = model.get("Latency (TTFT)", "")

            if model_name and (throughput or latency):
                performance_data.append({"model": model_name, "throughput": throughput, "latency": latency})

        save_dir = os.path.dirname(json_path)
        performance_path = os.path.join(save_dir, "llm_vellu_ai_performance_metrics.json")
        with open(performance_path, "w", encoding="utf-8") as f:
            json.dump(performance_data, f, indent=2, ensure_ascii=False)
        print(f"Saved performance metrics to: {performance_path}")

        return performance_data

    except Exception as e:
        print(f"Failed to analyze performance metrics: {str(e)}")
        return None


def main():
    url = "https://www.vellum.ai/llm-leaderboard"
    current_dir = os.path.dirname(os.path.abspath(__file__))

    data = fetch_leaderboard_data(url)
    if data:
        save_data(data, current_dir)

        model_details_path = os.path.join(current_dir, "leaderboard", "llm_vellu_ai_model_details.json")
        if os.path.exists(model_details_path):
            analyze_performance_metrics(model_details_path)


if __name__ == "__main__":
    main()
