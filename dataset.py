import pandas
import os


def get_df(file_name) -> pandas.DataFrame:
    return pandas.read_json(file_name)


def getddos() -> pandas.DataFrame:
    return get_df("alerts_ddos.json")


def getdistributeddenialofservice() -> pandas.DataFrame:
    return get_df("alerts_distributed_denial_of_service.json")


def getlongdups() -> pandas.DataFrame:
    return get_df("01-long-dupsremoved.json")


def getCombined():
    return get_df("alerts_combined.json")


def combinedatasets(df1, df2):
    return df1.append(df2, ignore_index=True)


def writecombineddatasets():
    pandas.DataFrame.to_json(combinedatasets(getddos(), getdistributeddenialofservice()), "alerts_combined.json")


def writedatasetodisk(dataset, filename):
    try:
        os.remove(filename)
    except OSError:
        pass
    dataset.to_json(filename, orient="columns")
