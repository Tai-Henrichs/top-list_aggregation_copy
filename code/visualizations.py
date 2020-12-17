import numpyfy
import search
import numpy as np
import plotnine as plt
import pandas as pd
from adjustText import adjust_text

dpi = 300
fileType = ".jpeg"
syntheticDirectory = "../Synthetic-Results/"
realDirectory = "../Real-World-Results/"

def paretoFilterDict(data):
    nonDominated = dict()

    for algo, info in data.items():
        dist, time = info
        dominated = False
        for _ , info2 in data.items():
            dist2, time2 = info2
            # > / >= because having a higher 
            # time or distance is worse
            if time >= time2 and dist >= dist2:
                if time > time2 or dist > dist2:
                    dominated = True
        if not dominated:
            nonDominated[algo] = (dist, time)
            
    return nonDominated

def infoForAlg(data):
        totalAccuracy = 0
        totalTime = 0
        numPoints = 0

        for row in data:
            totalAccuracy += row[1]
            totalTime += row[2]
            numPoints += 1.0

        return (totalAccuracy / numPoints, totalTime / numPoints)

def paretoByAllData():
    s = search.Search(syntheticDirectory)
    averages = dict()

    for algo in numpyfy.algorithms:
        _, fileName = s.filter_by_algo(algo)
        data = np.genfromtxt(fileName, delimiter = ",", skip_header = 1)
        averages[algo] = infoForAlg(data)
    return paretoFilterDict(averages)

def barChart(x, y, groupBy, df):
    plot =  (plt.ggplot(data=df) + 
        plt.aes(x=x, y=y, fill=f"factor({groupBy})") + 
            plt.geom_bar(stat = "identity") + 
                    plt.theme(axis_text_x=plt.element_blank()) + 
                        plt.labs(fill = f"{groupBy}")
    )
    return plot

def scatterPlot(x, y, groupBy, df):
    expansion = (2,2.2)
    adjusttext_settings = {'arrowprops': {
                                'arrowstyle': '->',
                                'color': 'black'
                            }, 'expand_text' :  expansion, 
                                'expand_points' : expansion, 
                                'expand_objects' : expansion}
    plot =  (plt.ggplot(data=df) + 
        plt.aes(x=x, y=y) + 
            plt.geom_point(mapping = 
                plt.aes(color=f"factor({groupBy})", 
                    shape=f"factor({groupBy})")) + 
                plt.labs(color = groupBy, shape = groupBy) + 
                    plt.geom_text(plt.aes(label=f"factor({groupBy})"), 
                                    size = 8,
                                    adjust_text = adjusttext_settings)
    )
    
    return plot

def paretoFilterDf(df):
    rows, columns = df.shape
    dominatedAlgorithms = list()

    for i in range(rows):
        dominated = False

        for j in range(rows):
            numWorse = 0
            numBetter = 0

            for k in range(1, columns):
                if df.iloc[i,k] > df.iloc[j,k]:
                    numWorse += 1
                elif df.iloc[i,k] < df.iloc[j,k]:
                    numBetter += 1 
                    break

            if  numWorse > 0 and numBetter == 0:
                    dominated = True
                    break

        if dominated:
            algorithm = df.iloc[i,0]
            dominatedAlgorithms.append(algorithm)
    
    for algorithm in dominatedAlgorithms:
        df = df[df.Algorithms != algorithm]

    return df
    

def normalizedData(df):
    minScore = df.min()[1]
    df['DISTANCE'] = df['DISTANCE'] / minScore 
    df.columns = ['Algorithms','Cost', 'Time (CPU Seconds)']
    return df

def dictToDfFormat(data):
        out = list()

        for algorithm, info in data.items():
            dist, time = info
            out.append([algorithm, dist, time])
        return out

def collapseDuplicates(df):
    # Value is a tuple: totalDist, totalTime, number of data-points
    includedAlgorithms = dict()
    
    rows, _ = df.shape 

    for i in range(0,rows):
        algorithm = df.iloc[i,0]

        if algorithm not in includedAlgorithms:
            includedAlgorithms[algorithm] = (df.iloc[i,1], df.iloc[i,2], 1)
        else:
            dist, time, num = includedAlgorithms[algorithm]

            distTotal = dist +  df.iloc[i,1]
            timeTotal = time +  df.iloc[i,2]
        
            includedAlgorithms[algorithm] = (distTotal, timeTotal, num + 1)

    toConvert = dict()
    for algorithm, info in includedAlgorithms.items():
        dist, time, num = info 
        # To ensure float division
        num = float(num)
        toConvert[algorithm] = (dist / num, time / num)
    
    header = df.columns.values.tolist()
    return pd.DataFrame(dictToDfFormat(toConvert), columns=header)


def fileComparison(fileName, folder):
    # Handles spaces
    sep="\\s*,\\s*"
    engine = "python"
    data = pd.read_csv(f"{folder}{fileName}", sep=sep, engine=engine)
    data = normalizedData(data)
    data = collapseDuplicates(data)
    data = paretoFilterDf(data)

    x = "Cost"
    y = "Time (CPU Seconds)"
    groupBy = "Algorithms"
    label = fileName[:len(fileName) - 4]

    outputLabel = f"{label}{fileType}"
    scatterPlot(x, y, groupBy, data).save(f"scatter_{outputLabel}")
    
    barChart(groupBy, x, groupBy, data).save(f"{x}_bar_{outputLabel}")
    barChart(groupBy, y, groupBy, data).save(f"{y}_bar_{outputLabel}")


def overallComparison():
    paretoOptimal = paretoByAllData()
    data = pd.DataFrame(dictToDfFormat(paretoOptimal), columns=["ALGORITHM", "DISTANCE", "TIME"])
    data = normalizedData(data)

    x = "Average Cost"
    y = "Average Time (CPU Seconds)"
    groupBy = "Algorithms"
    data.columns = [groupBy, x, y]

    scatterPlot(x,y,groupBy,data).save(f"Overall Algorithm Comparison{fileType}")


def perfByParamPlot(parameter):
    data = numpyfy.tidydf(parameter)

    # Convert to costs normalized by optimal
    minScore = data.min()[2]
    data['Average Kendall-Tau Distance'] = data['Average Kendall-Tau Distance'] / minScore
    data.columns = ["Algorithms", parameter, "Average Cost", "Average Time (CPU Seconds)"]

    paramValues = numpyfy.listByParam(parameter)

    # Scatter Plots for algorithm comparison
    for value in paramValues:
        if parameter == "n":
            toPlot = data[data.n == value]
        elif parameter == "N":
            toPlot = data[data.N == value]
        elif parameter == "th":
            toPlot = data[data.th == value]
        elif parameter == "k":
            toPlot = data[data.k == value]

        # Pareto Filtering
        toPlot = toPlot.drop(columns=[parameter])
        toPlot = paretoFilterDf(toPlot)

        plot = scatterPlot("Average Cost", "Average Time (CPU Seconds)", "Algorithms", toPlot)
        plot.save(f"{parameter}-{value}_scatter{fileType}")


if __name__ == '__main__':
    """ overallComparison()
    syntheticFiles = ["mallows_topk_n50_N5000_th0.001_k45.csv",
                    "mallows_topk_n30_N500_th0.01_k15.csv",
                        "mallows_topk_n10_N50_th0.1_k2.csv"]
    for name in syntheticFiles:
        fileComparison(name, syntheticDirectory)

    realFiles = ["CED-00010-00000046.csv",
                                "CED-00010-00000050.csv"]
                            
    for name in realFiles:
        fileComparison(name, realDirectory) """

    for param in ["n", "N", "th", "k"]:
        perfByParamPlot(param)