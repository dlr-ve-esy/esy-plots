import os

import pandas as pd
import pytest

from data.preparation import column_metadata, Metadatum, DataPreparationException, DataPreparer


def test__column_metadata__with_label__returns():
    result = column_metadata(label="MyLabel")
    assert Metadatum.Label in result.keys()
    assert result[Metadatum.Label] == "MyLabel"


def test__column_metadata__no_unit__returns_empty_unit():
    result = column_metadata(label="MyLabel")
    assert Metadatum.Label in result.keys()
    assert result[Metadatum.Unit] == ""


@pytest.mark.parametrize("label", [None, "", " "])
def test__column_metadata__invalid_label__fails(label):
    with pytest.raises(DataPreparationException):
        # noinspection PyTypeChecker
        column_metadata(label=label)


def test__column_metadata__with_label_and_unit__returns_both():
    result = column_metadata(label="ALabel", unit="m/s")
    assert Metadatum.Label in result.keys()
    assert result[Metadatum.Label] == "ALabel"
    assert Metadatum.Unit in result.keys()
    assert result[Metadatum.Unit] == "m/s"


def test__column_metadata__int_label__returns_converted():
    # noinspection PyTypeChecker
    result = column_metadata(label=4, unit="m/s")
    assert result[Metadatum.Label] == "4"


class TestDataPreparer:
    def test__save_to_file__empty__creates_file(self):
        path = "./deleteFile.hdf5"
        preparer = DataPreparer()
        preparer.save_to_file(path)
        assert os.path.isfile(path)
        os.remove(path)

    def test__save_to_file__file_extension_missing__appends(self):
        path = "./deleteFile"
        preparer = DataPreparer()
        preparer.save_to_file(path)
        assert os.path.isfile(f"{path}.hdf5")
        os.remove(f"{path}.hdf5")

    def test__save_to_file__valid_file_extension__no_append(self):
        for ext in ["h5", "hdf5", "he5"]:
            path = f"./deleteFile.{ext}"
            preparer = DataPreparer()
            preparer.save_to_file(path)
            assert os.path.isfile(path)
            os.remove(path)

    @pytest.mark.parametrize("group", [None, "", " "])
    def test__init_data_group__group_key_invalid__fails(self, group):
        preparer = DataPreparer()
        with pytest.raises(DataPreparationException):
            # noinspection PyTypeChecker
            preparer.init_data_group(group=None, key_metadata={})

    def test__init_data_group__no_keys__fails(self):
        preparer = DataPreparer()
        with pytest.raises(DataPreparationException):
            preparer.init_data_group(group="MyGroup", key_metadata={})

    @pytest.mark.parametrize("metadata", [{None: None}, {"": None}, {"  ": None}])
    def test__init_data_group__column_name_invalid__fails(self, metadata):
        preparer = DataPreparer()
        with pytest.raises(DataPreparationException):
            # noinspection PyTypeChecker
            preparer.init_data_group(group="MyGroup", key_metadata=metadata)

    @pytest.mark.parametrize("metadata", [{"Col": {Metadatum.Unit: "MyUnit"}},
                                          {"Col": {"label": "SomeLabel"}},
                                          {"Col": {Metadatum.Label: "Label", "unknown_key": "a"}}])
    def test__init_data_group__column_metadata_invalid__fails(self, metadata):
        preparer = DataPreparer()
        with pytest.raises(DataPreparationException):
            # noinspection PyTypeChecker
            preparer.init_data_group(group="MyGroup", key_metadata=metadata)

    def test__add_values__group_missing__fails(self):
        preparer = DataPreparer()
        with pytest.raises(DataPreparationException):
            preparer.add_values(group="MissingGroup", series=pd.Series(), metadata={})

    def test__add_values__index_missing__fails(self):
        preparer = DataPreparer()
        preparer.init_data_group(group="Group", key_metadata={"ColA": column_metadata(label="A")})
        with pytest.raises(DataPreparationException):
            preparer.add_values(group="Group", series=pd.Series([5, 2, 1]), metadata={})

    def test__add_values__index_name_mismatch__fails(self):
        preparer = DataPreparer()
        preparer.init_data_group(group="Group", key_metadata={"ColA": column_metadata(label="A")})
        series = pd.Series(data=[0], index=[0])
        series.index.name = "ColB"
        with pytest.raises(DataPreparationException):
            preparer.add_values(group="Group", series=series, metadata={})

    def test__add_values__index_length_mismatch__fails(self):
        preparer = DataPreparer()
        preparer.init_data_group(group="Group", key_metadata={"ColA": column_metadata(label="A"),
                                                              "ColB": column_metadata(label="B")})
        series = pd.Series(data=[0], index=[0])
        series.index.name = "ColA"
        with pytest.raises(DataPreparationException):
            preparer.add_values(group="Group", series=series, metadata={})

    def test__add_values__more_than_one_column__fails(self):
        preparer = DataPreparer()
        preparer.init_data_group(group="Group", key_metadata={"ColA": column_metadata(label="A")})
        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]}, index=[0, 1])
        df.index.name = "ColA"
        with pytest.raises(DataPreparationException):
            # noinspection PyTypeChecker
            preparer.add_values(group="Group", series=df, metadata=column_metadata(label="1"))

    def test__add_values__one_column_dataframe__accept(self):
        preparer = DataPreparer()
        preparer.init_data_group(group="Group", key_metadata={"ColA": column_metadata(label="A")})
        df = pd.DataFrame({'col1': [3, 4]}, index=[0, 1])
        df.index.name = "ColA"
        # noinspection PyTypeChecker
        preparer.add_values(group="Group", series=df, metadata=column_metadata(label="1"))

    @pytest.mark.parametrize("metadata", [{}, {"Col": {Metadatum.Unit: "MyUnit"}},
                                          {"Col": {"label": "SomeLabel"}},
                                          {"Col": {Metadatum.Label: "Label", "unknown_key": "a"}}])
    def test__add_values__new_column_invalid_metadata__fails(self, metadata):
        preparer = DataPreparer()
        preparer.init_data_group(group="Group", key_metadata={"ColA": column_metadata(label="A")})
        series = pd.Series(data=[0], index=[0])
        series.index.name = "ColA"
        with pytest.raises(DataPreparationException):
            # noinspection PyTypeChecker
            preparer.add_values(group="Group", series=series, metadata=metadata)

    def test__add_values__existing_column__metadata_ignored(self):
        preparer = DataPreparer()
        preparer.init_data_group(group="Group", key_metadata={"ColA": column_metadata(label="A")})
        series = pd.Series(data=[0], index=[0])
        series.index.name = "ColA"
        series.name = "MyValueColumn"
        preparer.add_values(group="Group", series=series, metadata=column_metadata(label="SomeLabel"))
        series2 = pd.Series(data=[1], index=[1])
        series2.index.name = "ColA"
        series2.name = "MyValueColumn"
        preparer.add_values(group="Group", series=series2, metadata={})
