install.packages("rsample")
install.packages("randomForest")
install.packages("ranger")
install.packages("caret")
install.packages("h2o")
library(rsample)      # data splitting 
library(randomForest) # basic implementation
library(ranger)       # a faster implementation of randomForest
library(caret)        # an aggregator package for performing many machine learning models
library(h2o)          # an extremely fast java-based platform
library(Metrics)
HouseDataTrain <- read.csv("C:/Users/Vincent Chui/Desktop/RStudio/Kaggle Competitions/House Prices - Advanced Regression Techniques/Data/train.csv")

#cleaning the data

#street type, is it paved or not. 1 = Paved, 0 = not paved
table(HouseDataTrain$Street)
HouseDataTrain$paved[HouseDataTrain$Street == "Pave"] <- 1
HouseDataTrain$paved[HouseDataTrain$Street != "Pave"] <- 0

#Zoning, residential or not. 1 = residential, 0 = non-residential
#A	Agriculture
#C	Commercial
#FV	Floating Village Residential
#I	Industrial
#RH	Residential High Density
#RL	Residential Low Density
#RP	Residential Low Density Park 
#RM	Residential Medium Density
table(HouseDataTrain$MSZoning)
price <- summarize(group_by(HouseDataTrain, MSZoning),
                   mean(SalePrice, na.rm=T))

HouseDataTrain$zone[HouseDataTrain$MSZoning %in% c("FV")] <- 4
HouseDataTrain$zone[HouseDataTrain$MSZoning %in% c("RL")] <- 3
HouseDataTrain$zone[HouseDataTrain$MSZoning %in% c("RH","RM")] <- 2
HouseDataTrain$zone[HouseDataTrain$MSZoning %in% c("C (all)")] <- 1

#Is there an alley or not. 1 = has alley(grvl or paved), 0 = no alley
table(HouseDataTrain$Alley)

price <- summarize(group_by(HouseDataTrain, Alley),
                   mean(SalePrice, na.rm=T))
HouseDataTrain$alleypave[HouseDataTrain$Alley %in% c("Pave")] <- 1
HouseDataTrain$alleypave[!HouseDataTrain$Alley %in% c("Pave")] <- 0

#if the house lot is regular shaped, then = 1, otherwise = 0
table(HouseDataTrain$LotShape)
HouseDataTrain$regshape[HouseDataTrain$LotShape == "Reg"] <- 1
HouseDataTrain$regshape[HouseDataTrain$LotShape != "Reg"] <- 0

#is the property on level ground or not. If yes, then = 1, otherwise = 0
table(HouseDataTrain$LandContour)
HouseDataTrain$flat[HouseDataTrain$LandContour == "Lvl"] <- 1
HouseDataTrain$flat[HouseDataTrain$LandContour != "Lvl"] <- 0

#All public utilites or not. If yes, then = 1, otherwise = 0 
table(HouseDataTrain$Utilities)
HouseDataTrain$pubutil[HouseDataTrain$Utilities == "AllPub"] <- 1
HouseDataTrain$pubutil[HouseDataTrain$Utilities != "AllPub"] <- 0

#slope of the property. If the slope is gentle then = 1 otherwise = 0 
HouseDataTrain$gentle_slope[HouseDataTrain$LandSlope == "Gtl"] <- 1
HouseDataTrain$gentle_slope[HouseDataTrain$LandSlope != "Gtl"] <- 0

#if the house in dead end or frontage on 3 sides of property then = 1, otherwise 0
table(HouseDataTrain$LotConfig)
HouseDataTrain$culdesac_fr3[HouseDataTrain$LotConfig %in% c("CulDSac", "FR3")] <- 1
HouseDataTrain$culdesac_fr3[!HouseDataTrain$LotConfig %in% c("CulDSac", "FR3")] <- 0

#neighborhood pricing summary
nbhdprice <- summarize(group_by(HouseDataTrain, Neighborhood),
                       mean(SalePrice, na.rm=T))

nbhdprice_lo <- filter(nbhdprice, nbhdprice$`mean(SalePrice, na.rm = T)` < 140000)
nbhdprice_med <- filter(nbhdprice, nbhdprice$`mean(SalePrice, na.rm = T)` < 200000 &
                          nbhdprice$`mean(SalePrice, na.rm = T)` >= 140000 )
nbhdprice_hi <- filter(nbhdprice, nbhdprice$`mean(SalePrice, na.rm = T)` >= 200000)

HouseDataTrain$nbhd_price_level[HouseDataTrain$Neighborhood %in% nbhdprice_lo$Neighborhood] <- 1
HouseDataTrain$nbhd_price_level[HouseDataTrain$Neighborhood %in% nbhdprice_med$Neighborhood] <- 2
HouseDataTrain$nbhd_price_level[HouseDataTrain$Neighborhood %in% nbhdprice_hi$Neighborhood] <- 3

#Condition 1. Positive feature = 1 otherwise = 0
table(HouseDataTrain$Condition1)
HouseDataTrain$pos_features_1[HouseDataTrain$Condition1 %in% c("PosA", "PosN")] <- 1
HouseDataTrain$pos_features_1[!HouseDataTrain$Condition1 %in% c("PosA", "PosN")] <- 0

#Condition 2. Positive feature = 1 otherwise = 0
table(HouseDataTrain$Condition2)
HouseDataTrain$pos_features_2[HouseDataTrain$Condition2 %in% c("PosA", "PosN")] <- 1
HouseDataTrain$pos_features_2[!HouseDataTrain$Condition2 %in% c("PosA", "PosN")] <- 0

#Building type. Detached vs non. If detached, =1, otherwise =0. We are considering end unit to be detached. 
table(HouseDataTrain$BldgType)
HouseDataTrain$twnhs_end_or_1fam[HouseDataTrain$BldgType %in% c("1Fam", "TwnhsE")] <- 1
HouseDataTrain$twnhs_end_or_1fam[!HouseDataTrain$BldgType %in% c("1Fam", "TwnhsE")] <- 0

#House style and price. filter the house style by price. Then categorize them in 1 ,2, 3
housestyle_price <- summarize(group_by(HouseDataTrain, HouseStyle),
                              mean(SalePrice, na.rm=T))

housestyle_lo <- filter(housestyle_price, housestyle_price$`mean(SalePrice, na.rm = T)` < 140000)
housestyle_med <- filter(housestyle_price, housestyle_price$`mean(SalePrice, na.rm = T)` < 200000 &
                           housestyle_price$`mean(SalePrice, na.rm = T)` >= 140000 )
housestyle_hi <- filter(housestyle_price, housestyle_price$`mean(SalePrice, na.rm = T)` >= 200000)

HouseDataTrain$house_style_level[HouseDataTrain$HouseStyle %in% housestyle_lo$HouseStyle] <- 1
HouseDataTrain$house_style_level[HouseDataTrain$HouseStyle %in% housestyle_med$HouseStyle] <- 2
HouseDataTrain$house_style_level[HouseDataTrain$HouseStyle %in% housestyle_hi$HouseStyle] <- 3

#Roof style and price
roofstyle_price <- summarize(group_by(HouseDataTrain, RoofStyle),
                             mean(SalePrice, na.rm=T))
HouseDataTrain$roof_hip_shed[HouseDataTrain$RoofStyle %in% c("Hip", "Shed")] <- 1
HouseDataTrain$roof_hip_shed[!HouseDataTrain$RoofStyle %in% c("Hip", "Shed")] <- 0

#Roof material and price
roofmatl_price <- summarize(group_by(HouseDataTrain, RoofMatl),
                            mean(SalePrice, na.rm=T))
train$roof_matl_hi[train$RoofMatl %in% c("Membran", "WdShake", "WdShngl")] <- 1
train$roof_matl_hi[!train$RoofMatl %in% c("Membran", "WdShake", "WdShngl")] <- 0

# Exterior covering on the house and price category
priceext <- summarize(group_by(HouseDataTrain, Exterior1st),
                      mean(SalePrice, na.rm=T))

matl_lo_1 <- filter(priceext, priceext$`mean(SalePrice, na.rm = T)` < 140000)
matl_med_1<- filter(priceext, priceext$`mean(SalePrice, na.rm = T)` < 200000 &
                      priceext$`mean(SalePrice, na.rm = T)` >= 140000 )
matl_hi_1 <- filter(priceext, priceext$`mean(SalePrice, na.rm = T)` >= 200000)

HouseDataTrain$exterior_1[HouseDataTrain$Exterior1st %in% matl_lo_1$Exterior1st] <- 1
HouseDataTrain$exterior_1[HouseDataTrain$Exterior1st %in% matl_med_1$Exterior1st] <- 2
HouseDataTrain$exterior_1[HouseDataTrain$Exterior1st %in% matl_hi_1$Exterior1st] <- 3

# Exterior 2 covering on the house and price category
priceext2 <- summarize(group_by(HouseDataTrain, Exterior2nd),
                       mean(SalePrice, na.rm=T))

matl_lo <- filter(priceext2, priceext2$`mean(SalePrice, na.rm = T)` < 140000)
matl_med <- filter(priceext2, priceext2$`mean(SalePrice, na.rm = T)` < 200000 &
                     priceext2$`mean(SalePrice, na.rm = T)` >= 140000 )
matl_hi <- filter(priceext2, priceext2$`mean(SalePrice, na.rm = T)` >= 200000)

HouseDataTrain$exterior_2[HouseDataTrain$Exterior2nd %in% matl_lo$Exterior2nd] <- 1
HouseDataTrain$exterior_2[HouseDataTrain$Exterior2nd %in% matl_med$Exterior2nd] <- 2
HouseDataTrain$exterior_2[HouseDataTrain$Exterior2nd %in% matl_hi$Exterior2nd] <- 3

#Masonry veneer type and price category
priceMasVnr <- summarize(group_by(HouseDataTrain, MasVnrType),
                         mean(SalePrice, na.rm=T))

HouseDataTrain$exterior_mason_1[HouseDataTrain$MasVnrType %in% c("Stone", "BrkFace") | is.na(HouseDataTrain$MasVnrType)] <- 1
HouseDataTrain$exterior_mason_1[!HouseDataTrain$MasVnrType %in% c("Stone", "BrkFace") & !is.na(HouseDataTrain$MasVnrType)] <- 0

#Exterior Quality
priceEXTQ <- summarize(group_by(HouseDataTrain, ExterQual),
                       mean(SalePrice, na.rm=T))

HouseDataTrain$exterior_cond[HouseDataTrain$ExterQual == "Ex"] <- 4
HouseDataTrain$exterior_cond[HouseDataTrain$ExterQual == "Gd"] <- 3
HouseDataTrain$exterior_cond[HouseDataTrain$ExterQual == "TA"] <- 2
HouseDataTrain$exterior_cond[HouseDataTrain$ExterQual == "Fa"] <- 1

#Exterior Condition
priceEXTC <- summarize(group_by(HouseDataTrain, ExterCond),
                       mean(SalePrice, na.rm=T))

HouseDataTrain$exterior_cond2[HouseDataTrain$ExterCond == "Ex"] <- 5
HouseDataTrain$exterior_cond2[HouseDataTrain$ExterCond == "Gd"] <- 4
HouseDataTrain$exterior_cond2[HouseDataTrain$ExterCond == "TA"] <- 3
HouseDataTrain$exterior_cond2[HouseDataTrain$ExterCond == "Fa"] <- 2
HouseDataTrain$exterior_cond2[HouseDataTrain$ExterCond == "Po"] <- 1

#Foundation 
priceFoundation <- summarize(group_by(HouseDataTrain, Foundation),
                             mean(SalePrice, na.rm=T))

HouseDataTrain$found_concrete[HouseDataTrain$Foundation == "PConc"] <- 1
HouseDataTrain$found_concrete[HouseDataTrain$Foundation != "PConc"] <- 0

#basement quality
priceBsmtQual <- summarize(group_by(HouseDataTrain, BsmtQual),
                           mean(SalePrice, na.rm=T))

HouseDataTrain$bsmt_cond1[HouseDataTrain$BsmtQual == "Ex"] <- 5
HouseDataTrain$bsmt_cond1[HouseDataTrain$BsmtQual == "Gd"] <- 4
HouseDataTrain$bsmt_cond1[HouseDataTrain$BsmtQual == "TA"] <- 3
HouseDataTrain$bsmt_cond1[HouseDataTrain$BsmtQual == "Fa"] <- 2
HouseDataTrain$bsmt_cond1[is.na(HouseDataTrain$BsmtQual)] <- 1

#basement condition
priceBsmtCond <- summarize(group_by(HouseDataTrain, BsmtCond),
                           mean(SalePrice, na.rm=T))

HouseDataTrain$bsmt_cond2[HouseDataTrain$BsmtCond == "Gd"] <- 5
HouseDataTrain$bsmt_cond2[HouseDataTrain$BsmtCond == "TA"] <- 4
HouseDataTrain$bsmt_cond2[HouseDataTrain$BsmtCond == "Fa"] <- 3
HouseDataTrain$bsmt_cond2[is.na(HouseDataTrain$BsmtCond)] <- 2
HouseDataTrain$bsmt_cond2[HouseDataTrain$BsmtCond == "Po"] <- 1

#Basement exposure
priceBsmtExp <- summarize(group_by(HouseDataTrain, BsmtExposure),
                          mean(SalePrice, na.rm=T))

HouseDataTrain$bsmt_exp[HouseDataTrain$BsmtExposure == "Gd"] <- 5
HouseDataTrain$bsmt_exp[HouseDataTrain$BsmtExposure == "Av"] <- 4
HouseDataTrain$bsmt_exp[HouseDataTrain$BsmtExposure == "Mn"] <- 3
HouseDataTrain$bsmt_exp[HouseDataTrain$BsmtExposure == "No"] <- 2
HouseDataTrain$bsmt_exp[is.na(HouseDataTrain$BsmtExposure)] <- 1

#Basement Finish Type
priceBsmtFin1 <- summarize(group_by(HouseDataTrain, BsmtFinType1),
                           mean(SalePrice, na.rm=T))

HouseDataTrain$bsmt_fin1[HouseDataTrain$BsmtFinType1 == "GLQ"] <- 5
HouseDataTrain$bsmt_fin1[HouseDataTrain$BsmtFinType1 == "Unf"] <- 4
HouseDataTrain$bsmt_fin1[HouseDataTrain$BsmtFinType1 == "ALQ"] <- 3
HouseDataTrain$bsmt_fin1[HouseDataTrain$BsmtFinType1 %in% c("BLQ", "Rec", "LwQ")] <- 2
HouseDataTrain$bsmt_fin1[is.na(HouseDataTrain$BsmtFinType1)] <- 1


#Basement Finish Type 2
priceBsmtFin2 <- summarize(group_by(HouseDataTrain, BsmtFinType2),
                           mean(SalePrice, na.rm=T))

HouseDataTrain$bsmt_fin2[HouseDataTrain$BsmtFinType2 == "ALQ"] <- 6
HouseDataTrain$bsmt_fin2[HouseDataTrain$BsmtFinType2 == "Unf"] <- 5
HouseDataTrain$bsmt_fin2[HouseDataTrain$BsmtFinType2 == "GLQ"] <- 4
HouseDataTrain$bsmt_fin2[HouseDataTrain$BsmtFinType2 %in% c("Rec", "LwQ")] <- 3
HouseDataTrain$bsmt_fin2[HouseDataTrain$BsmtFinType2 == "BLQ"] <- 2
HouseDataTrain$bsmt_fin2[is.na(HouseDataTrain$BsmtFinType2)] <- 1

#Heating Type
priceHeat <- summarize(group_by(HouseDataTrain, Heating),
                       mean(SalePrice, na.rm=T))


HouseDataTrain$gasheat[HouseDataTrain$Heating %in% c("GasA", "GasW")] <- 1
HouseDataTrain$gasheat[!HouseDataTrain$Heating %in% c("GasA", "GasW")] <- 0

#Heating COndition/Quality
priceHeatQC <- summarize(group_by(HouseDataTrain, HeatingQC),
                         mean(SalePrice, na.rm=T))

HouseDataTrain$heatqual[HouseDataTrain$HeatingQC == "Ex"] <- 5
HouseDataTrain$heatqual[HouseDataTrain$HeatingQC == "Gd"] <- 4
HouseDataTrain$heatqual[HouseDataTrain$HeatingQC == "TA"] <- 3
HouseDataTrain$heatqual[HouseDataTrain$HeatingQC == "Fa"] <- 2
HouseDataTrain$heatqual[HouseDataTrain$HeatingQC == "Po"] <- 1

#Central AC
priceAC <- summarize(group_by(HouseDataTrain, CentralAir),
                     mean(SalePrice, na.rm=T))

HouseDataTrain$air[HouseDataTrain$CentralAir == "Y"] <- 1
HouseDataTrain$air[HouseDataTrain$CentralAir == "N"] <- 0

#Electrical System Type
priceElec <- summarize(group_by(HouseDataTrain, Electrical),
                       mean(SalePrice, na.rm=T))

HouseDataTrain$standard_electric[HouseDataTrain$Electrical == "SBrkr" | is.na(HouseDataTrain$Electrical)] <- 1
HouseDataTrain$standard_electric[!HouseDataTrain$Electrical == "SBrkr" & !is.na(HouseDataTrain$Electrical)] <- 0

#Kitchen Quality
priceKitchQual <- summarize(group_by(HouseDataTrain, KitchenQual),
                            mean(SalePrice, na.rm=T))

HouseDataTrain$kitchen[HouseDataTrain$KitchenQual == "Ex"] <- 4
HouseDataTrain$kitchen[HouseDataTrain$KitchenQual == "Gd"] <- 3
HouseDataTrain$kitchen[HouseDataTrain$KitchenQual == "TA"] <- 2
HouseDataTrain$kitchen[HouseDataTrain$KitchenQual == "Fa"] <- 1

#Fireplace Quality
priceFireQu <- summarize(group_by(HouseDataTrain, FireplaceQu),
                         mean(SalePrice, na.rm=T))

HouseDataTrain$fire[HouseDataTrain$FireplaceQu == "Ex"] <- 5
HouseDataTrain$fire[HouseDataTrain$FireplaceQu == "Gd"] <- 4
HouseDataTrain$fire[HouseDataTrain$FireplaceQu == "TA"] <- 3
HouseDataTrain$fire[HouseDataTrain$FireplaceQu == "Fa"] <- 2
HouseDataTrain$fire[HouseDataTrain$FireplaceQu == "Po" | is.na(HouseDataTrain$FireplaceQu)] <- 1

#Garage Type
priceGarage <- summarize(group_by(HouseDataTrain, GarageType),
                         mean(SalePrice, na.rm=T))

HouseDataTrain$gar_attach[HouseDataTrain$GarageType %in% c("Attchd", "BuiltIn")] <- 1
HouseDataTrain$gar_attach[!HouseDataTrain$GarageType %in% c("Attchd", "BuiltIn")] <- 0

#garage finish
priceGarageFin <- summarize(group_by(HouseDataTrain, GarageFinish),
                            mean(SalePrice, na.rm=T))

HouseDataTrain$gar_finish[HouseDataTrain$GarageFinish %in% c("Fin", "RFn")] <- 1
HouseDataTrain$gar_finish[!HouseDataTrain$GarageFinish %in% c("Fin", "RFn")] <- 0

#garage Quality
priceGaragQC <- summarize(group_by(HouseDataTrain, GarageQual),
                          mean(SalePrice, na.rm=T))

HouseDataTrain$garqual[HouseDataTrain$GarageQual == "Ex"] <- 5
HouseDataTrain$garqual[HouseDataTrain$GarageQual == "Gd"] <- 4
HouseDataTrain$garqual[HouseDataTrain$GarageQual == "TA"] <- 3
HouseDataTrain$garqual[HouseDataTrain$GarageQual == "Fa"] <- 2
HouseDataTrain$garqual[HouseDataTrain$GarageQual == "Po" | is.na(HouseDataTrain$GarageQual)] <- 1

#Garage Condition
priceGaragCond <- summarize(group_by(HouseDataTrain, GarageCond),
                            mean(SalePrice, na.rm=T))

HouseDataTrain$garqual2[HouseDataTrain$GarageCond == "Ex"] <- 5
HouseDataTrain$garqual2[HouseDataTrain$GarageCond == "Gd"] <- 4
HouseDataTrain$garqual2[HouseDataTrain$GarageCond == "TA"] <- 3
HouseDataTrain$garqual2[HouseDataTrain$GarageCond == "Fa"] <- 2
HouseDataTrain$garqual2[HouseDataTrain$GarageCond == "Po" | is.na(HouseDataTrain$GarageCond)] <- 1

#Driveway 
priceDrive <- summarize(group_by(HouseDataTrain, PavedDrive),
                        mean(SalePrice, na.rm=T))

HouseDataTrain$paved_drive[HouseDataTrain$PavedDrive == "Y"] <- 1
HouseDataTrain$paved_drive[HouseDataTrain$PavedDrive != "Y"] <- 0
HouseDataTrain$paved_drive[is.na(HouseDataTrain$paved_drive)] <- 0

#Home Functionality
price <- summarize(group_by(HouseDataTrain, Functional),
                   mean(SalePrice, na.rm=T))

HouseDataTrain$housefunction[HouseDataTrain$Functional %in% c("Typ", "Min1", "Min2", "Mod")] <- 1
HouseDataTrain$housefunction[!HouseDataTrain$Functional %in% c("Typ", "Min1", "Min2", "Mod")] <- 0

#Pool Quality
pricePoolQC <- summarize(group_by(HouseDataTrain, PoolQC),
                         mean(SalePrice, na.rm=T))

HouseDataTrain$pool_good[HouseDataTrain$PoolQC %in% c("Ex")] <- 1
HouseDataTrain$pool_good[!HouseDataTrain$PoolQC %in% c("Ex")] <- 0

#Fence Quality
priceFencQC <- summarize(group_by(HouseDataTrain, Fence),
                         mean(SalePrice, na.rm=T))

HouseDataTrain$priv_fence[HouseDataTrain$Fence %in% c("GdPrv")] <- 1
HouseDataTrain$priv_fence[!HouseDataTrain$Fence %in% c("GdPrv")] <- 0

#This doesn't seem worth using at the moment. May adjust later.
priceMisc <- summarize(group_by(HouseDataTrain, MiscFeature),
                       mean(SalePrice, na.rm=T))


#Sale Type
price <- summarize(group_by(HouseDataTrain, SaleType),
                   mean(SalePrice, na.rm=T))

HouseDataTrain$sale_cat[HouseDataTrain$SaleType %in% c("New", "Con")] <- 5
HouseDataTrain$sale_cat[HouseDataTrain$SaleType %in% c("CWD", "ConLI")] <- 4
HouseDataTrain$sale_cat[HouseDataTrain$SaleType %in% c("WD")] <- 3
HouseDataTrain$sale_cat[HouseDataTrain$SaleType %in% c("COD", "ConLw", "ConLD")] <- 2
HouseDataTrain$sale_cat[HouseDataTrain$SaleType %in% c("Oth")] <- 1

#Condition of sale
priceSaleCond <- summarize(group_by(HouseDataTrain, SaleCondition),
                           mean(SalePrice, na.rm=T))

HouseDataTrain$sale_cond[HouseDataTrain$SaleCondition %in% c("Partial")] <- 4
HouseDataTrain$sale_cond[HouseDataTrain$SaleCondition %in% c("Normal", "Alloca")] <- 3
HouseDataTrain$sale_cond[HouseDataTrain$SaleCondition %in% c("Family","Abnorml")] <- 2
HouseDataTrain$sale_cond[HouseDataTrain$SaleCondition %in% c("AdjLand")] <- 1

#drop off the variables that have been made numeric and are no longer needed.
HouseDataTrain$Street <- NULL
HouseDataTrain$LotShape <- NULL
HouseDataTrain$LandContour <- NULL
HouseDataTrain$Utilities <- NULL
HouseDataTrain$LotConfig <- NULL
HouseDataTrain$LandSlope <- NULL
HouseDataTrain$Neighborhood <- NULL
HouseDataTrain$Condition1 <- NULL
HouseDataTrain$Condition2 <- NULL
HouseDataTrain$BldgType <- NULL
HouseDataTrain$HouseStyle <- NULL
HouseDataTrain$RoofStyle <- NULL
HouseDataTrain$RoofMatl <- NULL

HouseDataTrain$Exterior1st <- NULL
HouseDataTrain$Exterior2nd <- NULL
HouseDataTrain$MasVnrType <- NULL
HouseDataTrain$ExterQual <- NULL
HouseDataTrain$ExterCond <- NULL

HouseDataTrain$Foundation <- NULL
HouseDataTrain$BsmtQual <- NULL
HouseDataTrain$BsmtCond <- NULL
HouseDataTrain$BsmtExposure <- NULL
HouseDataTrain$BsmtFinType1 <- NULL
HouseDataTrain$BsmtFinType2 <- NULL

HouseDataTrain$Heating <- NULL
HouseDataTrain$HeatingQC <- NULL
HouseDataTrain$CentralAir <- NULL
HouseDataTrain$Electrical <- NULL
HouseDataTrain$KitchenQual <- NULL
HouseDataTrain$FireplaceQu <- NULL

HouseDataTrain$GarageType <- NULL
HouseDataTrain$GarageFinish <- NULL
HouseDataTrain$GarageQual <- NULL
HouseDataTrain$GarageCond <- NULL
HouseDataTrain$PavedDrive <- NULL

HouseDataTrain$Functional <- NULL
HouseDataTrain$PoolQC <- NULL
HouseDataTrain$Fence <- NULL
HouseDataTrain$MiscFeature <- NULL
HouseDataTrain$SaleType <- NULL
HouseDataTrain$SaleCondition <- NULL
HouseDataTrain$MSZoning <- NULL
HouseDataTrain$Alley <- NULL

#Remove NA
HouseDataTrain$GarageYrBlt[is.na(HouseDataTrain$GarageYrBlt)] <- 0
HouseDataTrain$MasVnrArea[is.na(HouseDataTrain$MasVnrArea)] <- 0
HouseDataTrain$LotFrontage[is.na(HouseDataTrain$LotFrontage)] <- 0

#Split the data

HouseDataTrain_Split <- initial_split(HouseDataTrain, prop = .7)
house_train <- training(HouseDataTrain_Split)
house_test <- testing(HouseDataTrain_Split)

#Basic Random Forest model

# for reproduciblity
set.seed(123)

# default RF model
m1 <- randomForest(
  formula = SalePrice ~ .,
  data    = house_train
)

#Plot the model
plot(m1)

#find number of trees with lowest MSE
which.min(m1$mse)

# RMSE of this optimal random forest
sqrt(m1$mse[which.min(m1$mse)])

#This means that the number of trees providing the lowest error rate is 288, providing avg home sale price error of $30,246

#=================================================================================================================
# create training and validation data 
set.seed(123)
valid_split <- initial_split(house_train, .8)

# training data
house_train_v2 <- analysis(valid_split)

# validation data
house_valid <- assessment(valid_split)
x_test <- house_valid[setdiff(names(house_valid), "SalePrice")]
y_test <- house_valid$SalePrice

rf_oob_comp <- randomForest(
  formula = SalePrice ~ .,
  data    = house_train_v2,
  xtest   = x_test,
  ytest   = y_test
)

# extract OOB & validation errors
oob <- sqrt(rf_oob_comp$mse)
validation <- sqrt(rf_oob_comp$test$mse)

# compare error rates
tibble::tibble(
  `Out of Bag Error` = oob,
  `Test error` = validation,
  ntrees = 1:rf_oob_comp$ntree
) %>%
  gather(Metric, RMSE, -ntrees) %>%
  ggplot(aes(ntrees, RMSE, color = Metric)) +
  geom_line() +
  scale_y_continuous(labels = scales::dollar) +
  xlab("Number of trees")

#Tuning the RF model
# names of features
features <- setdiff(names(house_train), "SalePrice")
set.seed(123)

m2 <- tuneRF(
  x          = house_train[features],
  y          = house_train$SalePrice,
  ntreeTry   = 500,
  mtryStart  = 5,
  stepFactor = 1.5,
  improve    = 0.01,
  trace      = TRUE      # False to not show real-time progress 
)

#full grid search across several hyper parameters

# RF speed
system.time(
  house_randomForest <- randomForest(
    formula = SalePrice ~ ., 
    data    = house_train, 
    ntree   = 500,
    mtry    = floor(length(features) / 3)
  )
)

# ranger speed (much faster option)
system.time(
  house_ranger <- ranger(
    formula   = SalePrice ~ ., 
    data      = house_train, 
    num.trees = 500,
    mtry      = floor(length(features) / 3)
  )
)

#hypergrid
hyper_grid <- expand.grid(
  mtry       = seq(20, 30, by = 2),
  node_size  = seq(3, 9, by = 2),
  sampe_size = c(.55, .632, .70, .80),
  OOB_RMSE   = 0
)

nrow(hyper_grid)


for(i in 1:nrow(hyper_grid)) {
  
  # train model
  model <- ranger(
    formula         = SalePrice ~ ., 
    data            = house_train, 
    num.trees       = 500,
    mtry            = hyper_grid$mtry[i],
    min.node.size   = hyper_grid$node_size[i],
    sample.fraction = hyper_grid$sampe_size[i],
    seed            = 123
  )
  
  # add OOB error to grid
  hyper_grid$OOB_RMSE[i] <- sqrt(model$prediction.error)
}

hyper_grid %>% 
  dplyr::arrange(OOB_RMSE) %>%
  head(10)

#it seems like the best option is mtry = 30, node_size = 3, sample size = 0.8, RSME ~ 28,896

#Rerun with optimal values
OOB_RMSE <- vector(mode = "numeric", length = 100)

for(i in seq_along(OOB_RMSE)) {
  
  optimal_ranger <- ranger(
    formula         = SalePrice ~ ., 
    data            = house_train, 
    num.trees       = 500,
    mtry            = 30,
    min.node.size   = 3,
    sample.fraction = .8,
    importance      = 'impurity'
  )
  
  OOB_RMSE[i] <- sqrt(optimal_ranger$prediction.error)
}

hist(OOB_RMSE, breaks = 20)

#We see that our expected error ranges between ~28,600-29,600 with a most likely just shy of 29,200.

#we set importance = 'impurity' in the above modeling, which allows us to assess variable importance. 
#Variable importance is measured by recording the decrease in MSE each time a variable is used as a node 
#split in a tree. The remaining error left in predictive accuracy after a node split is known as node 
#impurity and a variable that reduces this impurity is considered more imporant than those variables that
#do not. Consequently, we accumulate the reduction in MSE for each variable across all the trees and the 
#variable with the greatest accumulated impact is considered the more important, or impactful. 

#See the top 25 variables of importance
optimal_ranger$variable.importance %>% 
  tidy() %>%
  dplyr::arrange(desc(x)) %>%
  dplyr::top_n(25) %>%
  ggplot(aes(reorder(names, x), x)) +
  geom_col() +
  coord_flip() +
  ggtitle("Top 25 important variables")

#h2o full grid search - every combination of hyperparameter settings.

h2o.no_progress()
h2o.init(max_mem_size = "5g")

# create feature names
y <- "SalePrice"
x <- setdiff(names(house_train), y)

# turn training set into h2o object
train.h2o <- as.h2o(house_train)

# hyperparameter grid
hyper_grid.h2o <- list(
  ntrees      = seq(200, 500, by = 100),
  mtries      = seq(20, 30, by = 2),
  sample_rate = c(.55, .632, .70, .80)
)

# build grid search 
grid <- h2o.grid(
  algorithm = "randomForest",
  grid_id = "rf_grid",
  x = x, 
  y = y, 
  training_frame = train.h2o,
  hyper_params = hyper_grid.h2o,
  search_criteria = list(strategy = "Cartesian")
)

# collect the results and sort by our model performance metric of choice
grid_perf <- h2o.getGrid(
  grid_id = "rf_grid", 
  sort_by = "mse", 
  decreasing = FALSE
)
print(grid_perf)

# hyperparameter grid (random discrete )
hyper_grid.h2o <- list(
  ntrees      = seq(200, 500, by = 150),
  mtries      = seq(15, 35, by = 10),
  max_depth   = seq(20, 40, by = 5),
  min_rows    = seq(1, 5, by = 2),
  nbins       = seq(10, 30, by = 5),
  sample_rate = c(.55, .632, .75)
)

# random grid search criteria
search_criteria <- list(
  strategy = "RandomDiscrete",
  stopping_metric = "mse",
  stopping_tolerance = 0.005,
  stopping_rounds = 10,
  max_runtime_secs = 30*60
)

# build grid search 
random_grid <- h2o.grid(
  algorithm = "randomForest",
  grid_id = "rf_grid2",
  x = x, 
  y = y, 
  training_frame = train.h2o,
  hyper_params = hyper_grid.h2o,
  search_criteria = search_criteria
)

# collect the results and sort by our model performance metric of choice
grid_perf2 <- h2o.getGrid(
  grid_id = "rf_grid2", 
  sort_by = "mse", 
  decreasing = FALSE
)
print(grid_perf2)


# Grab the model_id for the top model, chosen by validation error
best_model_id <- grid_perf2@model_ids[[1]]
best_model <- h2o.getModel(best_model_id)
# Now let's evaluate the model performance on a test set
house_test.h2o <- as.h2o(house_test)
best_model_perf <- h2o.performance(model = best_model, newdata = house_test.h2o)

# RMSE of best model
h2o.mse(best_model_perf) %>% sqrt()

# randomForest
pred_randomForest <- predict(house_randomForest, house_test)
head(pred_randomForest)
model_output <- cbind(house_test, pred_randomForest)
rmse(model_output$SalePrice,model_output$pred_randomForest)

# ranger
pred_ranger <- predict(house_ranger, house_test)
head(pred_ranger$predictions)
model_output2 <- cbind(house_test, pred_ranger)

#gradient boosting machines

install.packages("gbm")
install.packages("xgboost")
install.packages("pdp")
install.packages("lime")

library(gbm)          # basic implementation
library(xgboost)      # a faster implementation of gbm
library(pdp)          # model visualization
library(ggplot2)      # model visualization
library(lime)         # model visualization

# for reproducibility
set.seed(123)

# train GBM model
gbm.fit <- gbm(
  formula = SalePrice ~ .,
  distribution = "gaussian",
  data = house_train,
  n.trees = 10000,
  interaction.depth = 1,
  shrinkage = 0.001,
  cv.folds = 5,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)  

# print results
print(gbm.fit)

# get MSE and compute RMSE
sqrt(min(gbm.fit$cv.error))


# plot loss function as a result of n trees added to the ensemble
gbm.perf(gbm.fit, method = "cv")

#This shows that the minimum CV RMSE is 34010. This means on average our model is about $34,010 off the actual sales price

#Tuning the model

# create hyperparameter grid
hyper_grid <- expand.grid(
  shrinkage = c(.01, .1, .3),
  interaction.depth = c(1, 3, 5),
  n.minobsinnode = c(5, 10, 15),
  bag.fraction = c(.65, .8, 1), 
  optimal_trees = 0,               # a place to dump results
  min_RMSE = 0                     # a place to dump results
)

# total number of combinations
nrow(hyper_grid)

# randomize data
random_index <- sample(1:nrow(house_train), nrow(house_train))
random_house_train <- house_train[random_index, ]

# grid search 
for(i in 1:nrow(hyper_grid)) {
  
  # reproducibility
  set.seed(123)
  
  # train model
  gbm.tune <- gbm(
    formula = SalePrice ~ .,
    distribution = "gaussian",
    data = random_house_train,
    n.trees = 5000,
    interaction.depth = hyper_grid$interaction.depth[i],
    shrinkage = hyper_grid$shrinkage[i],
    n.minobsinnode = hyper_grid$n.minobsinnode[i],
    bag.fraction = hyper_grid$bag.fraction[i],
    train.fraction = .75,
    n.cores = NULL, # will use all cores by default
    verbose = FALSE
  )
  
  # add min training error and trees to grid
  hyper_grid$optimal_trees[i] <- which.min(gbm.tune$valid.error)
  hyper_grid$min_RMSE[i] <- sqrt(min(gbm.tune$valid.error))
}

hyper_grid %>% 
  dplyr::arrange(min_RMSE) %>%
  head(10)

# modify hyperparameter grid
hyper_grid <- expand.grid(
  shrinkage = c(.01, .05, .1),
  interaction.depth = c(3, 5, 7),
  n.minobsinnode = c(5, 7, 10),
  bag.fraction = c(.65, .8, 1), 
  optimal_trees = 0,               # a place to dump results
  min_RMSE = 0                     # a place to dump results
)

# total number of combinations
nrow(hyper_grid)


# grid search 
for(i in 1:nrow(hyper_grid)) {
  
  # reproducibility
  set.seed(123)
  
  # train model
  gbm.tune <- gbm(
    formula = SalePrice ~ .,
    distribution = "gaussian",
    data = random_house_train,
    n.trees = 2200,
    interaction.depth = hyper_grid$interaction.depth[i],
    shrinkage = hyper_grid$shrinkage[i],
    n.minobsinnode = hyper_grid$n.minobsinnode[i],
    bag.fraction = hyper_grid$bag.fraction[i],
    train.fraction = .75,
    n.cores = NULL, # will use all cores by default
    verbose = FALSE
  )
  
  # add min training error and trees to grid
  hyper_grid$optimal_trees[i] <- which.min(gbm.tune$valid.error)
  hyper_grid$min_RMSE[i] <- sqrt(min(gbm.tune$valid.error))
}

hyper_grid %>% 
  dplyr::arrange(min_RMSE) %>%
  head(10)

# for reproducibility
set.seed(123)

# train GBM model
gbm.fit.final <- gbm(
  formula = SalePrice ~ .,
  distribution = "gaussian",
  data = house_train,
  n.trees = 93,
  interaction.depth = 7,
  shrinkage = 0.1,
  n.minobsinnode = 7,
  bag.fraction = .65, 
  train.fraction = 1,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)  

#visualize which variables have the largest influence on sale price.
par(mar = c(5, 8, 1, 1))
summary(
  gbm.fit.final, 
  cBars = 20,
  method = relative.influence, # also can use permutation.test.gbm
  las = 2
)

#Partial dependence plots
gbm.fit.final %>%
  partial(pred.var = "GrLivArea", n.trees = gbm.fit.final$n.trees, grid.resolution = 100) %>%
  autoplot(rug = TRUE, train = house_train) +
  scale_y_continuous(labels = scales::dollar)

#PDP plot below displays the average change in predicted sales price as we vary GrLivArea while holding all other variables 
#constant. This is done by holding all variables constant for each observation in our training data set but then apply the 
#unique values of GrLivArea for each observation. We then average the sale price across all the observations. This PDP 
#illustrates how the predicted sales price increases as the square footage of the ground floor in a house increases.


#ICE Curve (individual conditional expectation)
ice1 <- gbm.fit.final %>%
  partial(
    pred.var = "GrLivArea", 
    n.trees = gbm.fit.final$n.trees, 
    grid.resolution = 100,
    ice = TRUE
  ) %>%
  autoplot(rug = TRUE, train = house_train, alpha = .1) +
  ggtitle("Non-centered") +
  scale_y_continuous(labels = scales::dollar)

ice2 <- gbm.fit.final %>%
  partial(
    pred.var = "GrLivArea", 
    n.trees = gbm.fit.final$n.trees, 
    grid.resolution = 100,
    ice = TRUE
  ) %>%
  autoplot(rug = TRUE, train = house_train, alpha = .1, center = TRUE) +
  ggtitle("Centered") +
  scale_y_continuous(labels = scales::dollar)

gridExtra::grid.arrange(ice1, ice2, nrow = 1)

#ICE curves are an extension of PDP plots but, rather than plot the average marginal effect on the response variable, we 
#plot the change in the predicted response variable for each observation as we vary each predictor variable.

# predict values for test data
pred <- predict(gbm.fit.final, n.trees = gbm.fit.final$n.trees, house_test)

# results
rmse(pred, house_test$Sale_Price)

#XGBoost Model - only works with matrices that contain all numeric variables

# variable names
features <- setdiff(names(house_train), "SalePrice")

# Create the treatment plan from the training data
treatplan <- vtreat::designTreatmentsZ(house_train, features, verbose = FALSE)

# Get the "clean" variable names from the scoreFrame
new_vars <- treatplan %>%
  magrittr::use_series(scoreFrame) %>%        
  dplyr::filter(code %in% c("clean", "lev")) %>% 
  magrittr::use_series(varName)     

# Prepare the training data
features_train <- vtreat::prepare(treatplan, house_train, varRestriction = new_vars) %>% as.matrix()
response_train <- house_train$SalePrice

# Prepare the test data
features_test <- vtreat::prepare(treatplan, house_test, varRestriction = new_vars) %>% as.matrix()
response_test <- house_test$SalePrice

# dimensions of one-hot encoded data
dim(features_train)
dim(features_test)

# reproducibility
set.seed(123)

xgb.fit1 <- xgb.cv(
  data = features_train,
  label = response_train,
  nrounds = 1000,
  nfold = 5,
  objective = "reg:linear",  # for regression models
  verbose = 0               # silent,
)

# get number of trees that minimize error
xgb.fit1$evaluation_log %>%
  dplyr::summarise(
    ntrees.train = which(train_rmse_mean == min(train_rmse_mean))[1],
    rmse.train   = min(train_rmse_mean),
    ntrees.test  = which(test_rmse_mean == min(test_rmse_mean))[1],
    rmse.test   = min(test_rmse_mean)
  )

#to identify the minimum RMSE and the optimal number of trees for both the training data and the 
#cross-validated error. We can see that the training error continues to decrease to 465 trees where the
#RMSE nearly reaches zero; however, the cross validated error reaches a minimum RMSE of $32,726 with only 28 trees.

# plot error vs number trees
ggplot(xgb.fit1$evaluation_log) +
  geom_line(aes(iter, train_rmse_mean), color = "red") +
  geom_line(aes(iter, test_rmse_mean), color = "blue")

#xgb.cv allows early stopping. Instructs the function to stop running if the cross validated error does not improve
#for n continuous trees. In this example it is 10.

# reproducibility
set.seed(123)

xgb.fit2 <- xgb.cv(
  data = features_train,
  label = response_train,
  nrounds = 1000,
  nfold = 5,
  objective = "reg:linear",  # for regression models
  verbose = 0,               # silent,
  early_stopping_rounds = 10 # stop if no improvement for 10 consecutive trees
)

# plot error vs number trees
ggplot(xgb.fit2$evaluation_log) +
  geom_line(aes(iter, train_rmse_mean), color = "red") +
  geom_line(aes(iter, test_rmse_mean), color = "blue")

#tuning the model

# create parameter list
params <- list(
  eta = .1,
  max_depth = 5,
  min_child_weight = 2,
  subsample = .8,
  colsample_bytree = .9
)


# reproducibility
set.seed(123)

# train model
xgb.fit3 <- xgb.cv(
  params = params,
  data = features_train,
  label = response_train,
  nrounds = 1000,
  nfold = 5,
  objective = "reg:linear",  # for regression models
  verbose = 0,               # silent,
  early_stopping_rounds = 10 # stop if no improvement for 10 consecutive trees
)

# assess results
xgb.fit3$evaluation_log %>%
  dplyr::summarise(
    ntrees.train = which(train_rmse_mean == min(train_rmse_mean))[1],
    rmse.train   = min(train_rmse_mean),
    ntrees.test  = which(test_rmse_mean == min(test_rmse_mean))[1],
    rmse.test   = min(test_rmse_mean)
  )

# create hyperparameter grid
hyper_grid <- expand.grid(
  eta = c(.01, .05, .1, .3),
  max_depth = c(1, 3, 5, 7),
  min_child_weight = c(1, 3, 5, 7),
  subsample = c(.65, .8, 1), 
  colsample_bytree = c(.8, .9, 1),
  optimal_trees = 0,               # a place to dump results
  min_RMSE = 0                     # a place to dump results
)

nrow(hyper_grid)


# grid search 
for(i in 1:nrow(hyper_grid)) {
  
  # create parameter list
  params <- list(
    eta = hyper_grid$eta[i],
    max_depth = hyper_grid$max_depth[i],
    min_child_weight = hyper_grid$min_child_weight[i],
    subsample = hyper_grid$subsample[i],
    colsample_bytree = hyper_grid$colsample_bytree[i]
  )
  
  # reproducibility
  set.seed(123)
  
  # train model
  xgb.tune <- xgb.cv(
    params = params,
    data = features_train,
    label = response_train,
    nrounds = 5000,
    nfold = 5,
    objective = "reg:linear",  # for regression models
    verbose = 0,               # silent,
    early_stopping_rounds = 10 # stop if no improvement for 10 consecutive trees
  )
  
  # add min training error and trees to grid
  hyper_grid$optimal_trees[i] <- which.min(xgb.tune$evaluation_log$test_rmse_mean)
  hyper_grid$min_RMSE[i] <- min(xgb.tune$evaluation_log$test_rmse_mean)
}

hyper_grid %>%
  dplyr::arrange(min_RMSE) %>%
  head(10)

# parameter list
params <- list(
  eta = 0.1,
  max_depth = 5,
  min_child_weight = 1,
  subsample = 0.65,
  colsample_bytree = 0.9
)

# train final model
xgb.fit.final <- xgboost(
  params = params,
  data = features_train,
  label = response_train,
  nrounds = 1576,
  objective = "reg:linear",
  verbose = 0
)

# create importance matrix
importance_matrix <- xgb.importance(model = xgb.fit.final)

# variable importance plot
xgb.plot.importance(importance_matrix, top_n = 10, measure = "Gain")

# predict values for test data
pred <- predict(xgb.fit.final, features_test)

# results
caret::RMSE(pred, response_test)
