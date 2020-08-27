import pandas as pd
import numpy as np
import Log as log
pd.options.mode.chained_assignment = None  # default='warn'

# scales all rows from a dataframe to be inside the range of 0 to 1
class Scaler:

    # creates a minmax list for each column of the dataframe
    def __init__(self, dataframe :pd.DataFrame):
        self.minmax = [] # Column number : {"max" : Max in column, "min" : Min in column}
        i = 0
        for col in dataframe.columns:
            self.minmax.append({})
            self.minmax[i]["max"] = max(dataframe[col].values)
            self.minmax[i]["min"] = min(dataframe[col].values)
            i += 1

    # in place transformation of given data, scales according to the minmax values
    def transform(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.astype(np.float32)
            i = 0
            for col in data.columns:
                minn = self.minmax[i]["min"]
                maxn = self.minmax[i]["max"]
                if maxn - minn != 0:
                    data[col] = (data[col] - minn) / (maxn - minn)
                i += 1
        else:
            for i in range(len(data)):
                for j in range(len(data[i])):
                    num = data[i][j]
                    minn = self.minmax[j]["min"]
                    maxn = self.minmax[j]["max"]
                    if maxn - minn != 0:
                        data[i][j] = (num - minn) / (maxn - minn)
        return data

# the base class for reading a csv file, provides a simple platform for getting results and can be furthered modified in the future
class Dataset:
    # df - mostly used by DatasetHandler, but provides a way to inject the dataframe directly into the class
    # resultsColumn - the name of the column inside the dataframe, where the labels of each row is located
    # dataTypes - a list with all the data types of each column inside the dataframe, used for better performance when reading large files (~1M rows)
    # numrows - number of rows that will be read from the file provided
    # skiprows - a list with all the number of rows that will not be included in the dataframe
    def __init__(self, df :pd.DataFrame = None, filename: str = None, resultsColumn: str = "", dataTypes :dict = None, numrows :int=None, skiprows=None):
        if filename is None:
            self.csv = df
        else:
            self.csv: pd.DataFrame = pd.read_csv(filename, dtype=dataTypes,
                                             nrows=numrows, skiprows=skiprows)
        self.results = self.csv[resultsColumn].tolist()
        self.csv.drop(resultsColumn, 1, inplace=True)

    def getResult(self, i):
        return self.results[i]
    
# Handles everything that has to do with csv files, and train-test dataframes
class DatasetHandler:

    # normalText - the label inside the results column that indicates if the row is benign or not
    # testingDS - provides a way to inject the testing dataframe directly, if it's not given, the testing dataset will be built automatically by choosing random rows from the csv file provided
    # trainingDS - provides a way to inject the testing dataframe directly, if it's not given, the training dataset will be built automatically by choosing random rows from the csv file provided
    # NOTE: if both testingDS and trainingDS are None, they will be both built from separate random rows found inside the csv file
    # droppedColumns - determines what columns should be removes from each row when given to the ML
    # trainAmount - a number from 0 to 1, indicates the portion of the training dataset, in terms of percentage 0.9 -> 90% of the file
    # dataTypes - a list with all the data types of each column inside the dataframe, used for better performance when reading large files (~1M rows)
    # numrows - number of rows that will be read from the file provided
    # skiprows - a list with all the number of rows that will not be included in the dataframe
    def __init__(self, filename: str, normalText :str, resultsColumn: str = "", testingDS :Dataset = None, trainingDS :Dataset = None, droppedColumns: list = [],
                 trainAmount = 1, dataTypes :dict = None, numrows :int=None, skiprows=None):
        if trainingDS is None or testingDS is None:
            self.csv: pd.DataFrame = pd.read_csv(filename, dtype=dataTypes,
                                             nrows=numrows, skiprows=skiprows)
        else:
            self.csv :pd.DataFrame = pd.read_csv(filename, dtype=dataTypes, nrows=2)
        log.show("datahandler", "csv created")
        self.normalText = normalText
        self.resultsColumn = resultsColumn
        self.droppedColumns = {} # Column name: Column index
        self.columns = self.csv.columns.tolist()
        for col in droppedColumns:
            self.droppedColumns[col] = self.columns.index(col)
        self.testingPrefix = int(round(len(self.csv) * trainAmount))
        self.trainingDF :Dataset = None
        self.testingDF :Dataset = None
        if testingDS is not None:
            self.testingDF = testingDS
        if trainingDS is not None:
            self.trainingDF = trainingDS
        self.scaler = None
        self.textConverter = {}  # Column name : Set of values
        self.__setup()

    # initial setup of all the class variables
    def __setup(self):
        self.__trainTestSplit()
        log.show("datahandler", "train test split")
        self.__convertText()
        log.show("datahandler", "csv converted")
        self.dataframe = self.csv.values.tolist()
        self.__prepareScaler()
        log.show("datahandler", "dataframe scaled")


    # constructs each of the dataframes in case one of them wasn't provided beforehand
    # shuffles the whole dataframe, and splits it between the testing and traing dataframes
    def __trainTestSplit(self):
        if self.testingDF is None:
            self.csv :pd.DataFrame = self.csv.sample(frac=1).reset_index(drop=True)
            if self.trainingDF is None:
                self.trainingDF = Dataset(df=self.csv.iloc[:self.testingPrefix, :], resultsColumn=self.resultsColumn)
            self.testingDF = Dataset(df=self.csv.iloc[self.testingPrefix:, :], resultsColumn=self.resultsColumn)
        else:
            if self.trainingDF is None:
                self.trainingDF = Dataset(self.csv, resultsColumn=self.resultsColumn)
        for col in self.droppedColumns.keys():
            self.trainingDF.csv.drop(col, 1, inplace=True)

    # converts all text values to float numeric ones and creates a text converter for future use
    def __convertText(self):
        for column in self.trainingDF.csv.columns:
            if self.trainingDF.csv[column].dtype != np.float32 and self.trainingDF.csv[column].dtype != np.float64 and self.trainingDF.csv[column].dtype != np.int32 and self.trainingDF.csv[column].dtype != np.int64:
                log.show("datahandler", f"converting column {column} from type: {self.csv[column].dtype}")
                self.textConverter[column] = {}
                i = 0
                for c in set(self.trainingDF.csv[column].values.tolist()):
                    self.textConverter[column][c] = i
                    i += 1
                currentConvert = self.textConverter[column]
                for i in range(len(self.trainingDF.csv[column])):
                    self.trainingDF.csv[column][i] = currentConvert[self.trainingDF.csv[column][i]]

    # scales all rows inside the training dataframe to be inside the range of 0 to 1, and saves the scaler for future use
    def __prepareScaler(self):
        self.scaler = Scaler(self.trainingDF.csv)
        self.trainingDF.csv = self.scaler.transform(self.trainingDF.csv)

    # returns a prepared for ML used copy of the data received, by removing unneeded values, scaling the rest, and converting string columns to numeric ones
    def prepareData(self, data: list) -> list:
        newdata = data.copy()
        for index in sorted(self.droppedColumns.values(), reverse=True):
            del newdata[index]
        for column in self.textConverter:
            columnloc = self.columns.index(column)
            if newdata[columnloc] not in self.textConverter[column].keys():
                self.textConverter[column][newdata[columnloc]] = max(self.textConverter[column].values()) + 1
            newdata[columnloc] = self.textConverter[column][newdata[columnloc]]
        newdata = self.scaler.transform([newdata])[0]
        return newdata

    # returns the result of an already known data, by receiving it's index, and where is it located, on the TestingDataset or the Training one
    def getResult(self, i, testing=False) -> str:
        if testing:
            return self.testingDF.getResult(i)
        else:
            return self.trainingDF.getResult(i)

    # a more simple way to check if an already known data is benign or not, receiving the index of the data, and where is it located, on the TestingDataset or the Training one
    def isNormal(self, i, testing=False):
        return self.getResult(i, testing) == self.normalText

    # a generic approach for getting a certain dataframe
    def getData(self, testing=False):
        if testing:
            return self.getTestingData()
        else:
            return self.getTrainingData()

    # returns the training dataframe in a list form, should only be used by the system itself
    def getTrainingData(self):
        return self.trainingDF.csv.values.tolist()

    # returns the training dataframe in a list form, should only be used by the system itself
    def getTestingData(self):
        return self.testingDF.csv.values.tolist()

