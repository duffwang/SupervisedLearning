#Cleaning Data
#Original dataset found at https://archive.ics.uci.edu/ml/datasets/Phishing+Websites
x <- read.csv('PhishingWebsitesData.csv')
y <- na.omit(x)
z <- y[sample(nrow(y), 6000),]
rownames(z) <- NULL
z['Blank'] = 0
write.csv(z, 'phishing_clean.csv', row.names = F)
z = read.csv('phishing_clean.csv')
sum(z[['Result']] == 1)

x <- read.csv('winequality-data.csv')
sum(x[['quality']] >= 6)

#Clean new coders survey data
#Original data file can be found at https://www.kaggle.com/freecodecamp/2016-new-coder-survey-
x <- read.csv('2016-FCC-New-Coders-Survey-Data.csv')

x <- x[,c("Age","EmploymentStatus","HasFinancialDependents",
                                                      "HasHighSpdInternet","IsEthnicMinority",
                                                      "IsSoftwareDev","MonthsProgramming","SchoolDegree","HasDebt")]
y <- na.omit(x)
z <- y[sample(nrow(y), 6000),]
rownames(z) <- NULL

z[, 'EmploymentStatus'] <- as.integer(as.factor(z[, 'EmploymentStatus']))
z[, 'SchoolDegree'] <- as.integer(as.factor(z[, 'SchoolDegree']))


write.csv(z, 'codersurvey_clean.csv', row.names = F)
