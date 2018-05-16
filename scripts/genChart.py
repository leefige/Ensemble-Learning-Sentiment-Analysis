import re

logDir = "../log/"
chartDir = "../chart/"

def extractBag(name, ctype):
    path = logDir + "%s_%s.log" % (name, ctype)
    dstPath = chartDir + "%s_%s.md" % (name, ctype)
    with open(dstPath, 'w', encoding='utf8') as fout:
        fout.write("|No.|clf num|feature size|sample rate|RMSE|\n")
        fout.write("|-:|-:|-:|-:|-:|\n")
        with open(path, 'r', encoding='utf8') as fin:
            detp = re.compile(r"dtNum = (\d+), feature_num = (\d+), test_percent = ([\d\.]+), sample_size = ([\d\.]+)")
            rmsep = re.compile(r"Test RMSE: ([\d\.]+)")
            cnt = 0
            for line in fin:
                line = line.strip()
                dobj = detp.match(line)
                robj = rmsep.match(line)
                if dobj:
                    cnt += 1
                    dtnum = dobj.group(1)
                    featnum = dobj.group(2)
                    samplerate = dobj.group(4)
                    fout.write("|%d|%s|%s|%s|" % (cnt, dtnum, featnum, samplerate))
                elif robj:
                    rmse = robj.group(1)
                    fout.write("%s|\n" % (rmse))      

def extractBoost(name, ctype):
    path = logDir + "%s_%s.log" % (name, ctype)
    dstPath = chartDir + "%s_%s.md" % (name, ctype)
    with open(dstPath, 'w', encoding='utf8') as fout:
        fout.write("|No.|max iteration|real iteration|feature size|RMSE|\n")
        fout.write("|-:|-:|-:|-:|-:|\n")
        with open(path, 'r', encoding='utf8') as fin:
            detp = re.compile(r"iter = (\d+), feature_num = (\d+), test_percent = ([\d\.]+)")
            rmsep = re.compile(r"Test RMSE: ([\d\.]+)")
            cnt = 0
            for line in fin:
                line = line.strip()
                dobj = detp.match(line)
                robj = rmsep.match(line)
                if dobj:
                    cnt += 1
                    realiter = dobj.group(1)
                    featnum = dobj.group(2)
                    fout.write("|%d|0|%s|%s|" % (cnt, realiter, featnum))
                elif robj:
                    rmse = robj.group(1)
                    fout.write("%s|\n" % (rmse))

if __name__ == '__main__':
    extractBag("Bagging", "dtree")
    extractBag("Bagging", "svm")
    extractBoost("Boosting", "dtree")
    extractBoost("Boosting", "svm")
    