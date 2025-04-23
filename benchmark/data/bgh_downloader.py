# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Big Bench Hard Dataset"""

import json
import requests
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

all_configs = [
    'tracking_shuffled_objects_seven_objects',
    'salient_translation_error_detection',
    'tracking_shuffled_objects_three_objects',
    'geometric_shapes',
    'object_counting',
    'word_sorting',
    'logical_deduction_five_objects',
    'hyperbaton',
    'sports_understanding',
    'logical_deduction_seven_objects',
    'multistep_arithmetic_two',
    'ruin_names',
    'causal_judgement',
    'logical_deduction_three_objects',
    'formal_fallacies',
    'snarks',
    'boolean_expressions',
    'reasoning_about_colored_objects',
    'dyck_languages',
    'navigate',
    'disambiguation_qa',
    'temporal_sequences',
    'web_of_lies',
    'tracking_shuffled_objects_five_objects',
    'penguins_in_a_table',
    'movie_recommendation',
    'date_understanding'
]

_CITATION = """\
@article{suzgun2022challenging,
  title={Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them},
  author={Suzgun, Mirac and Scales, Nathan and Sch{\"a}rli, Nathanael and Gehrmann, Sebastian and Tay, Yi and Chung, Hyung Won and Chowdhery, Aakanksha and Le, Quoc V and Chi, Ed H and Zhou, Denny and and Wei, Jason},
  journal={arXiv preprint arXiv:2210.09261},
  year={2022}
}
"""

_DESCRIPTION = """\
BIG-Bench (Srivastava et al., 2022) is a diverse evaluation suite that focuses on tasks believed to be beyond the capabilities of current language models. Language models have already made good progress on this benchmark, with the best model in the BIG-Bench paper outperforming average reported human-rater results on 65% of the BIG-Bench tasks via few-shot prompting. But on what tasks do language models fall short of average human-rater performance, and are those tasks actually unsolvable by current language models?
"""

_HOMEPAGE = "https://github.com/suzgunmirac/BIG-Bench-Hard"

_LICENSE = "MIT license"

def download_datasets():
    # Create data directory if it doesn't exist
    data_dir = Path(__file__).parent
    bbh_dir = data_dir / "bbh"
    bbh_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup session with retry mechanism
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    # Download each config from GitHub
    base_url = "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh"
    for config in all_configs:
        print(f"Downloading {config}...")
        url = f"{base_url}/{config}.json"
        try:
            response = session.get(url, verify=False)
            if response.status_code == 200:
                data = response.json()
                output_file = bbh_dir / f"{config}.jsonl"
                with open(output_file, 'w', encoding='utf-8') as f:
                    for example in data['examples']:
                        f.write(json.dumps(example) + '\n')
                print(f"Saved {config} to {output_file}")
            else:
                print(f"Failed to download {config}: Status code {response.status_code}")
        except Exception as e:
            print(f"Error downloading {config}: {str(e)}")
            time.sleep(2)  # Wait before next attempt

if __name__ == "__main__":
    # Disable SSL verification warnings
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    download_datasets()