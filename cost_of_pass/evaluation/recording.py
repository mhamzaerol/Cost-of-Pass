import io
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, fields
from typing import List, Optional, Union

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, file_exists, hf_hub_download, repo_exists

from cost_of_pass.metrics.base import EvaluationMetric
from cost_of_pass.tasks.base import BaseQuery, FullQuery

from pathlib import Path


@dataclass(kw_only=True)
class BaseRecord:
    """
    This is the basic record structure that any run should at least produce and save to hub. 
    This is useful to calculate the costs of a run, and find the efficiency estimate.
    """
    model_name: str
    task_name: str
    tt_method_name: str
    input_idx: int
    answer: str
    num_input_tokens: int
    num_prompt_tokens: int
    num_completion_tokens: int
    num_answer_tokens: int
    cost_per_prompt_token: float
    cost_per_completion_token: float
    completed: Union[bool, None] = None
    timestamp: Union[float, None] = None
    uid: Union[str, None] = None

    def __post_init__(self):
        if self.uid is None:
            self.uid = str(uuid.uuid4())
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.completed is None:
            # check if the <incomplete>...<\incomplete> tag is present in the answer
            matches = re.findall(r"<incomplete>(.*?)</incomplete>", self.answer, re.DOTALL)
            self.completed = (len(matches) == 0)

    def as_base(self) -> "BaseRecord":
        """
        Extracts a BaseRecord instance from self by filtering only the fields
        defined in BaseRecord.
        """
        base_field_names = {f.name for f in fields(BaseRecord)}
        base_data = {name: getattr(self, name) for name in base_field_names}
        return BaseRecord(**base_data)
    
    @classmethod
    def from_base(cls, base: "BaseRecord", **kwargs) -> "BaseRecord":
        """
        Initializes an instance of cls (which may be BaseRecord or a subclass)
        using data from a BaseRecord instance. Additional fields can be provided via kwargs.
        """
        base_field_names = {f.name for f in fields(BaseRecord)}
        base_data = {name: getattr(base, name) for name in base_field_names}
        base_data.update(kwargs)
        return cls(**base_data)

    @property
    def cost(self) -> float:
        return self.practical_cost

    @property
    def practical_cost(self) -> float:
        return (self.cost_per_prompt_token * self.num_prompt_tokens \
                + self.cost_per_completion_token * self.num_completion_tokens)
    
    @property
    def theoretical_cost(self) -> float:
        return (self.cost_per_prompt_token * self.num_input_tokens \
                + self.cost_per_completion_token * self.num_completion_tokens)


@dataclass(kw_only=True)
class FullRecord(BaseRecord):
    """
    This has the full record structure that we would like to save to the hub.
    We do not need to access this everytime we wish to compute efficiency.
    However, saving as much information as possible is useful for debugging and analysis.
    """
    input: str
    target: str
    prompts: List[str]
    responses: List[str]
    metadata: dict = None

    @property
    def as_query(self) -> BaseQuery:
        return BaseQuery(
            idx=self.input_idx,
            input=self.input,
            target=self.target
        )

    @property
    def as_full_query(self) -> FullQuery:
        return FullQuery.from_base(self.as_query, metadata=self.metadata)


@dataclass(kw_only=True)
class MetricRecord(BaseRecord):
    """
    For a selected metric, we save the metric name and the associated score.
    The score could be accuracy, or another type of reward.
    """
    metric_name: str
    metric_score: float

    @property
    def performance(self) -> float:
        return self.metric_score


@dataclass(kw_only=True)
class DatasetMetadata:
    """
    Metadata for a dataset.
    """
    n_runs: int
    n_queries: int
    n_records: int
    n_completed: int

    @property
    def is_empty(self) -> bool:
        return all(value == 0 for value in vars(self).values())
    
    @property
    def is_valid(self) -> bool:
        return self.is_empty or all(value > 0 for value in vars(self).values())

    @classmethod
    def empty(cls) -> 'DatasetMetadata':
        return cls(n_runs=0, n_queries=0, n_records=0, n_completed=0)
    
    @classmethod
    def from_records(cls, records: List[BaseRecord]) -> 'DatasetMetadata':
        if not records:
            return cls.empty()

        n_records = len(records)
        unique_queries = {record.input_idx for record in records}
        n_queries = len(unique_queries)
        n_completed = sum(record.completed for record in records)

        if n_records % n_queries != 0:
            raise ValueError("Number of records must be divisible by the number of queries for consistent metadata.")

        return cls(n_runs=n_records // n_queries, n_queries=n_queries, n_records=n_records, n_completed=n_completed)


class HubManager:
    """
    Manager for handling record storage and metadata updates on the Hugging Face Hub.
    Provides abstractions for full, base, and metric records and their metadata.
    """
    def __init__(
        self, 
        model, 
        task, 
        tt_method, 
        org_id: str = "CostOfPass", 
        repo_id: str = "benchmark",
        hf_api: Optional[HfApi] = None,
        token: Optional[str] = None,
    ):
        """
        Initialize a HubManager instance.
        """
        self.model = model if isinstance(model, str) else model.model_name
        self.task = task if isinstance(task, str) else task.task_name
        self.tt_method = tt_method if isinstance(tt_method, str) else tt_method.method_name
        self.org_id = org_id
        self.repo_id = repo_id
        self._api = hf_api or HfApi()
        self._token = token or os.getenv("HF_TOKEN")
        
        if not self._token:
            raise ValueError("HF_TOKEN environment variable not set and no token provided")

    @property
    def full_repo_id(self) -> str:
        """Get the full repository ID including organization."""
        return f"{self.org_id}/{self.repo_id}"

    @property
    def base_path(self) -> str:
        """Get the base path for all records."""
        return f"{self.task}/{self.model}/{self.tt_method}"

    @property
    def full_records_path(self) -> str:
        """Get the path for full records."""
        return f"{self.base_path}/full_records"
    
    @property
    def overall_metadata_path(self) -> str:
        """Get the path for the overall metadata."""
        return f"{self.base_path}/metadata.json"

    def metric_records_path(self, metric: Union[str, EvaluationMetric]) -> str:
        """
        Get the path for metric records.
        """
        metric_name = metric if isinstance(metric, str) else metric.name
        return f"{self.base_path}/{metric_name}_records"

    def _ensure_repo_exists(self) -> None:
        """Ensure the repository exists, creating it if necessary."""
        try:
            if not repo_exists(self.full_repo_id, repo_type="dataset", token=self._token):
                print(f"Repository {self.full_repo_id} not found. Creating it under {self.org_id}...")
                self._api.create_repo(
                    repo_id=self.full_repo_id,
                    repo_type="dataset",
                    private=True,
                    token=self._token
                )
        except Exception as e:
            raise RuntimeError(f"Failed to ensure repository exists: {e}") from e

    def _file_exists(self, path: str) -> bool:
        """
        Check if a file exists in the repository.
        """
        try:
            return file_exists(self.full_repo_id, path, repo_type="dataset", token=self._token)
        except Exception as e:
            raise RuntimeError(f"Failed to check if file exists at {path}: {e}") from e

    def _delete_file(self, path: str) -> bool:
        """
        Delete a file from the repository.
        """
        try:
            if self._file_exists(path):
                self._api.delete_file(
                    path_in_repo=path,
                    repo_id=self.full_repo_id,
                    repo_type="dataset",
                    token=self._token
                )
                return True
            return False
        except Exception as e:
            raise RuntimeError(f"Failed to delete file at {path}: {e}") from e

    def load_overall_metadata(self) -> dict:
        """
        Load overall metadata from the Hugging Face Hub dataset repository.
        """
        if not self._file_exists(self.overall_metadata_path):
            print(f"Warning: Metadata file not found at {self.overall_metadata_path}.")
            return {}

        try:
            # Download metadata file from the repository
            local_path = hf_hub_download(
                repo_id=self.full_repo_id,
                filename=self.overall_metadata_path,
                repo_type="dataset",
                token=self._token,
            )

            # Load the JSON metadata
            with open(local_path, "r", encoding="utf-8") as f:
                overall_metadata = json.load(f)
            
            return overall_metadata if overall_metadata else {}
        except Exception as e:
            raise RuntimeError(f"Failed to load metadata from {self.overall_metadata_path}: {e}") from e
        
    def _save_overall_metadata(self, overall_metadata: dict) -> None:
        """
        Save overall metadata to the Hugging Face Hub.
        """
        try:
            # Serialize the metadata to JSON
            json_str = json.dumps(overall_metadata, indent=2).encode("utf-8")
            json_buffer = io.BytesIO(json_str)

            # Upload the JSON metadata file
            self._api.upload_file(
                path_or_fileobj=json_buffer,
                path_in_repo=self.overall_metadata_path,
                repo_id=self.full_repo_id,
                repo_type="dataset",
                token=self._token,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to save metadata: {e}") from e

    def load_dataset_metadata(self, record_type: str) -> DatasetMetadata:
        """
        Load metadata for a specific record type (dataset) from the overall metadata.
        """
        overall_metadata = self.load_overall_metadata()
        if record_type in overall_metadata:
            return DatasetMetadata(**overall_metadata[record_type])
        else:
            return DatasetMetadata.empty()

    def _load_records_only(self, path: str) -> List[BaseRecord]:
        """
        Load dataset records from the Hugging Face Hub repository.
        """
        dataset_path = f"{path}/dataset.parquet"

        if not self._file_exists(dataset_path):
            print(f"Warning: Dataset file not found at {dataset_path}.")
            return []

        try:
            # Download dataset file from the repository
            local_path = hf_hub_download(
                repo_id=self.full_repo_id,
                filename=dataset_path,
                repo_type="dataset",
                token=self._token,
            )

            # Load the dataset from the downloaded Parquet file
            ds = load_dataset("parquet", data_files={"records": local_path}, split="records")
            return self.dataset_to_records(ds)
        except Exception as e:
            raise RuntimeError(f"Failed to load records from {path}: {e}") from e

    def dataset_to_records(self, dataset: Dataset) -> List[BaseRecord]:
        """
        Convert a dataset to a list of records.
        """
        if not dataset or len(dataset) == 0:
            return []
            
        # Determine record class based on column presence
        if "metric_name" in dataset.column_names:
            record_cls = MetricRecord
        elif "input" in dataset.column_names:
            record_cls = FullRecord
        else:
            raise ValueError("Dataset does not contain expected columns for records.")
            
        return [record_cls(**record) for record in dataset]

    def records_to_dataset(self, records: List[BaseRecord]) -> Dataset:
        """
        Convert a list of records to a dataset.
        """
        if not records:
            return Dataset.from_dict({})
            
        keys = records[0].__dataclass_fields__.keys()
        data = {key: [getattr(record, key) for record in records] for key in keys}
        return Dataset.from_dict(data)

    def save_records(
        self, 
        path: str, 
        new_records: List[BaseRecord], 
        overwrite: bool = True
    ) -> None:
        """
        Save records to the Hugging Face Hub.
        """
        # Ensure repository exists
        self._ensure_repo_exists()
        
        # Load existing records if not overwriting
        existing_records = [] if overwrite else self._load_records_only(path)
        combined_records = existing_records + new_records
        
        dataset_path = f"{path}/dataset.parquet"
        
        # Handle empty combined_records: delete file and update metadata
        if not combined_records:
            if self._file_exists(dataset_path):
                self._delete_file(dataset_path)
                print(f"Deleted empty dataset at {dataset_path}.")
            self._update_dataset_metadata_for_path(path, combined_records)
        else:
            # Convert records to dataset
            dataset = self.records_to_dataset(combined_records)

            # Serialize the Dataset to an in-memory Parquet file
            parquet_buffer = io.BytesIO()
            dataset.to_parquet(parquet_buffer)
            parquet_buffer.seek(0)  # Reset buffer position to the beginning

            # Upload the Parquet file
            try:
                self._api.upload_file(
                    path_or_fileobj=parquet_buffer,
                    path_in_repo=dataset_path,
                    repo_id=self.full_repo_id,
                    repo_type="dataset",
                    token=self._token,
                )
                self._update_dataset_metadata_for_path(path, combined_records)
            except Exception as e:
                print(f"Failed to save records to {path}: {e}")
                
                cur_dir = (Path(__file__).parent).resolve()
                rand_dir_name = str(uuid.uuid4())
                save_dir = cur_dir / "failed_to_save" / f"{rand_dir_name}"
                save_dir.mkdir(parents=True, exist_ok=True)

                # save the dataset as a parquet file
                dataset_save_path = save_dir / f"dataset.parquet"
                dataset.to_parquet(dataset_save_path)
                    
                # save the metadata as a json file
                metadata_save_path = save_dir / f"metadata.json"
                metadata = DatasetMetadata.from_records(combined_records)
                metadata_dict = metadata.__dict__
                with open(metadata_save_path, "w") as f:
                    json.dump(metadata_dict, f, indent=2)
                
                # save path information too
                with open(save_dir / "hub_path_info.txt", "w") as f:
                    f.write(dataset_path)
                    f.write("\n")
                    f.write(self.full_repo_id)
                
                print(f"Saved dataset and metadata to {save_dir}. Please check the files, and read the README.md for more information.")
                raise RuntimeError(f"Failed to save records to {path}: {e}") from e
            
    def _update_dataset_metadata_for_path(
        self, 
        path: str, 
        new_records: List[BaseRecord],
    ) -> None:
        """
        Update metadata for a specific path.
        """
        try:
            # Extract record type from path
            record_type = path.split("/")[-1]
            
            # Create new metadata based
            new_dataset_metadata = DatasetMetadata.from_records(new_records)
                
            # Update metadata to reflect empty dataset
            overall_metadata = self.load_overall_metadata()
            if new_dataset_metadata.is_empty:
                if record_type in overall_metadata:
                    del overall_metadata[record_type]
            else:
                overall_metadata[record_type] = new_dataset_metadata.__dict__
            self._save_overall_metadata(overall_metadata)
            
        except Exception as e:
            raise RuntimeError(f"Failed to update metadata for {path}: {e}") from e

    def save_full_records(self, full_records: List[FullRecord], overwrite: bool = True) -> None:
        """
        Save full records and their derived base records.
        """
        try:
            # Save full records
            self.save_records(self.full_records_path, full_records, overwrite=overwrite)
            
            if overwrite:
                existing_metrics = self.list_available_metrics()
                for metric in existing_metrics:
                    self.save_records(self.metric_records_path(metric), [], overwrite=True)
        except Exception as e:
            raise RuntimeError(f"Failed to save full records: {e}") from e
    
    def save_metric_records(self, metric: Union[str, EvaluationMetric], metric_records: List[MetricRecord], overwrite: bool = True) -> None:
        """
        Save metric records.
        """
        try:
            self.save_records(self.metric_records_path(metric), metric_records, overwrite=overwrite)
        except Exception as e:
            raise RuntimeError(f"Failed to save metric records: {e}") from e
    
    @property
    def full_records(self) -> List[FullRecord]:
        """Get the full records from the repository."""
        return self._load_records_only(self.full_records_path)

    @property
    def full_records_metadata(self) -> DatasetMetadata:
        """Get the metadata for full records."""
        return self.load_dataset_metadata("full_records")

    def metric_records(self, metric: Union[str, EvaluationMetric]) -> List[MetricRecord]:
        """
        Get the metric records for a specific metric.
        """
        return self._load_records_only(self.metric_records_path(metric))

    def metric_records_metadata(self, metric: Union[str, EvaluationMetric]) -> DatasetMetadata:
        """
        Get the metadata for a specific metric.
        """
        record_type = self.metric_records_path(metric).split("/")[-1]
        return self.load_dataset_metadata(record_type)

    def list_available_metrics(self) -> List[str]:
        """
        List all available metrics in the repository.
        """
        overall_metadata = self.load_overall_metadata()
        return [
            key.replace("_records", "") 
            for key in overall_metadata.keys() 
            if key != "full_records"
        ]
