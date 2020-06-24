"""Custom CSV Logger
"""
import csv
import collections.abc
import pathlib
import torch
import pandas as pd


class SimpleCSVLogger:

    def __init__(self, file, columns):
        """General purpose CSV Logger

        Initialized with a set of columns, it then has two operations
          - set(**kwargs) - to add entries into the current row
          - update - flush a row to file

        Arguments:
            file {str} -- Path to file
            columns {List[str]} -- List of keys that CSV is going to log
        """

        self.file = open(file, 'w')
        self.columns = columns
        self.values = {}

        self.writer = csv.writer(self.file)
        self.writer.writerow(self.columns)
        self.file.flush()

    def set(self, **kwargs):
        """Set value for current row

        [description]

        Arguments:
            **kwargs {[type]} -- [description]

        Raises:
            ValueError -- [description]
        """
        for k, v in kwargs.items():
            if k in self.columns:
                if isinstance(v, torch.Tensor):
                    v = v.item()
                self.values[k] = v
            else:
                raise ValueError(f"{k} not in columns {self.columns}")

    def flush(self):
        """Take current values and write a row in the CSV
        """
        row = [self.values.get(c, "") for c in self.columns]
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        """Close the file descriptor for the CSV
        """
        self.file.close()


class CSVLogger:

    def __init__(self, file):
        """ More flexible version of a CSVLogger
        The main difference is that columns need not be specified on creation
        and can be added through the lifetime of the object
        """
        self.file = file
        self.columns = []
        self.values = [[]]

        if pathlib.Path(file).exists():
            df = pd.read_csv(file)
            self.columns = list(df.columns)
            self.values = df.values.tolist()
            self._new_row()

        self.new_columns = False

    def set(self, *args, **kwargs):

        for mapping in args:
            assert isinstance(mapping, collections.abc.Mapping)
            for k, v in mapping.items():
                assert k not in kwargs
                kwargs[k] = v

        for k, v in kwargs.items():
            if k not in self.columns:
                self.columns.append(k)
                self.new_columns = True

                for row in self.values:
                    row.append(None)

            i = self.columns.index(k)
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.values[-1][i] = v

    def flush(self):
        if self.new_columns:
            df = pd.DataFrame(data=self.values, columns=self.columns)
            df.to_csv(self.file, index=False)
            self.new_columns = False
        else:
            df = pd.DataFrame(data=[self.values[-1]], columns=self.columns)
            df.to_csv(self.file, mode='a', header=False, index=False)
        self._new_row()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.flush()

    def _new_row(self):
        empty_row = [None]*len(self.columns)
        self.values.append(empty_row)

