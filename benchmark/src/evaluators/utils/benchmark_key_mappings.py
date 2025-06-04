BENCHMARK_KEY_MAPPINGS = {
    "humaneval": {
        "id": "task_id",
        "problem": "prompt",
        "solution": "canonical_solution",
        "test": "test",
        "entry_point": "entry_point",
        "test_imports": None
    },
    "mbpp": {
        "id": "task_id",
        "problem": "prompt",
        "solution": "code",
        "test": "test",
        "entry_point": "entry_point",
        "test_imports": "test_imports"
    },
    "math": {
        "id": "id",
        "problem": "problem",
        "solution": "solution",
        "test": None,
        "entry_point": None,
        "test_imports": None
    },
    "bbh": {
        "id": "task_id",
        "problem": "input",
        "solution": "target",
        "test": None,
        "entry_point": None,
        "test_imports": None
    },
    "drop": {
        "id": "id",
        "problem": "context",
        "solution": "ref_text",
        "test": None,
        "entry_point": None,
        "test_imports": None,
        "instruction_id_list":None,
        "kwargs": None,
    },
    "ifeval": {
        "id": "key",
        "problem": "prompt",
        "solution": None,
        "test": None,
        "entry_point": None,
        "test_imports": None,
        "instruction_id_list": "instruction_id_list",
        "kwargs": "kwargs",
    }
} 