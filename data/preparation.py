from enum import Enum, auto
from json import dumps
from typing import Dict, Union

import pandas as pd


class DataPreparationException(Exception):
    """An error that occurred during data preparation"""
    pass


class _Type(Enum):
    Data = auto()
    Metadata = auto()


class Metadatum(Enum):
    """Defines which metadata exist for plot data columns"""
    Label = auto()
    Unit = auto()


def column_metadata(label: str, unit: str = None) -> Dict[Metadatum, str]:
    """
    Creates a metadata dictionary from given input

    Args:
        label: full label of the column - as to be displayed for axis descriptions
        unit: of the associated data

    Returns:
        dictionary of the metadata
    """
    return {Metadatum.Label: label, Metadatum.Unit: unit}


class DataPreparer:
    """Prepare data to be used in different types of plots"""

    def __init__(self) -> None:
        """Create a new DataPreparer"""
        self.datasets: Dict[str, Dict[_Type, Union[pd.DataFrame, Dict]]] = {}

    def save_to_file(self, out_file_path: str) -> None:
        """
        Write all data to given file in hdf5 format

        Args:
            out_file_path: name of file to write
        """
        if not any([extension in out_file_path for extension in ["h5", "hdf5", "he5"]]):
            out_file_path = f"{out_file_path}.hdf5"

        store = pd.HDFStore(path=out_file_path, mode="w")
        for key, item in self.datasets.items():
            store.put(key=key, value=self._group_by_index(item[_Type.Data]))
            metadata = self._convert_enums(item[_Type.Metadata])
            store.get_storer(key=key).attrs.plot_metadata = dumps(metadata, ensure_ascii=False).encode(
                'utf8')
        store.close()

    @staticmethod
    def _group_by_index(data: pd.DataFrame) -> pd.DataFrame:
        """
        Returns data grouped by its own index to remove empty columns

        Args:
            data: dataframe to be compressed

        Returns:
            compressed data with
        """
        return data.groupby(list(data.index.names)).sum()

    @staticmethod
    def _convert_enums(metadata: Dict[str, Dict[Enum, str]]) -> Dict[str, Dict[str, str]]:
        """
        Turn inner keys of a dictionary from enum into lower-case strings

        Args:
            metadata: dictionary with enum keys

        Returns:
            the same dictionary but with lower-case string keys instead of enums in its inner dictionaries
        """
        return {k: {enum.name.lower(): v for enum, v in entries.items()} for k, entries in metadata.items()}

    def init_data_group(self, group: str, key_metadata: Dict[str, Dict[Metadatum, str]]) -> None:
        """
        Initialise a new data group under given name and metadata for each key column

        Args:
            group: (unique) name for the new data group that is used to address the data during plotting
            key_metadata: metadata description of **all** key columns as dictionary of shape
                {column_name: dict_of_column_metadata}

        Raises:
            DataPreparationException: if group name already exists
        """
        if group in self.datasets.keys():
            raise DataPreparationException(f"Group name '{group}' already exists.")

        key_columns = [column_name for column_name in key_metadata.keys()]
        empty_df = pd.DataFrame(columns=key_columns)
        empty_df.set_index(key_columns, inplace=True)

        self.datasets[group] = {
            _Type.Data: empty_df,
            _Type.Metadata: key_metadata,
        }

    def add_value_column(self, group: str, column: pd.Series, metadata: Dict[Metadatum, str]) -> None:
        """
        Add a value column to an existing data group with the associated metadata

        Args:
            group: name of data group to add the column to
            column: data of the column - (multi)index must match that of the data group
            metadata: metadata description of the column

        Raises:
            DataPreparationException:
                * if group name was not yet initialised,
                * if column data index does not match that of the group
        """
        self._assert_group_name_exists(group)
        container = self.datasets[group][_Type.Data]
        self._assert_indexes_match(container, column)
        column = self._ensure_is_series(column)

        self.datasets[group][_Type.Data] = pd.concat([container, column], axis=1)
        self.datasets[group][_Type.Metadata].update({column.name: metadata})

    def _assert_group_name_exists(self, group) -> None:
        """
        Raises exception if group name does not yet exist in data sets

        Args:
            group: data group that is tested for existence

       Raises:
            DataPreparationException: if group name was not yet initialised,
        """
        if group not in self.datasets.keys():
            raise DataPreparationException(f"Group name '{group}' already exists.")

    @staticmethod
    def _assert_indexes_match(container: pd.DataFrame, column: pd.Series) -> None:
        """
        Raises exception if indexes of container and column do not match

        Args:
            container: defining the index
            column: to have the same index

        Raises:
            DataPreparationException: if column data index does not match that of the group
        """
        group_index = container.index.names
        column_index = column.index.names
        if len(group_index) != len(column_index):
            raise DataPreparationException(
                f"Length of index in column to add does not match that of the assigned data group.")
        for index_column in group_index:
            if index_column not in column_index:
                raise DataPreparationException(f"Index column '{index_column}' not found in index of column to add.")

    @staticmethod
    def _ensure_is_series(series: Union[pd.Series, pd.DataFrame]) -> pd.Series:
        """
        Ensures that given object is a Series

        Returns:
            given series or dataframe squeezed to series
        Raises:
            DataPreparationException: if given object is a DataFrame with more than one column
        """
        if isinstance(series, pd.DataFrame):
            if len(series.columns) > 1:
                raise DataPreparationException(f"Given data must be a Series of single-column DataFrame!")
            series = series.squeeze()
        return series

    def add_value_rows(self, group: str, rows: pd.Series, metadata: Dict[Metadatum, str] = None) -> None:
        """
        Add rows of values to a (new or existing) column in an existing data group

        Args:
            group: data group to add the rows to
            rows: rows for one column - (multi)index must match that of the data group
            metadata: metadata description of the column - only required if the column is not yet, otherwise ignored

        Raises:
            DataPreparationException:
                * if group name was not yet initialised,
                * if column data index does not match that of the group
                * if a new column is specified and metadata are missing
        """
        self._assert_group_name_exists(group)
        container = self.datasets[group][_Type.Data]
        self._assert_indexes_match(container, rows)
        rows = self._ensure_is_series(rows)

        if rows.name not in list(container.columns):
            if not metadata:
                raise DataPreparationException(f"No metadata specified for new column '{rows.name}'.")
            self.datasets[group][_Type.Metadata].update({rows.name: metadata})
        self.datasets[group][_Type.Data] = pd.concat([container, pd.DataFrame(rows)], axis=0)
