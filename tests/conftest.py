import pandas as pd
import pytest
from sklearn.datasets import load_diabetes, load_wine

from agentic_analyst.workspace import Workspace


@pytest.fixture
def wine_ws(tmp_path):
    df = load_wine(as_frame=True).frame.rename(columns={"target": "cultivar"})
    df["cultivar"] = df["cultivar"].map({0: "A", 1: "B", 2: "C"})
    return Workspace(df=df, name="wine", target="cultivar", out_dir=tmp_path)


@pytest.fixture
def diabetes_ws(tmp_path):
    df = load_diabetes(as_frame=True).frame.rename(columns={"target": "progression"})
    return Workspace(df=df, name="diabetes", target="progression", out_dir=tmp_path)


@pytest.fixture
def small_ws(tmp_path):
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, None, 6.0],
            "y": [2.1, 3.9, 6.2, 8.1, 9.8, 12.2],
            "group": ["a", "a", "b", "b", "c", "c"],
        }
    )
    return Workspace(df=df, name="small", target=None, out_dir=tmp_path)
