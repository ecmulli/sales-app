import <- function(){
      stores <- fread('store.csv')
      test <- fread('test.csv')
      train <- fread('train.csv')
      
      train <- merge(train, stores, by = 'Store')
      test <- merge(test, stores, by = 'Store')
      
      train[,':='(StoreType = as.factor(StoreType),
                  Assortment = as.factor(Assortment),
                  DayOfWeek = as.factor(DayOfWeek),
                  Open = as.factor(Open),
                  Promo = as.factor(Promo),
                  SchoolHoliday = as.factor(SchoolHoliday),
                  StateHoliday = as.factor(StateHoliday),
                  Date = as.Date(Date))]
      
      test[,':='(StoreType = as.factor(StoreType),
                  Assortment = as.factor(Assortment),
                  DayOfWeek = as.factor(DayOfWeek),
                  Open = as.factor(Open),
                  Promo = as.factor(Promo),
                  SchoolHoliday = as.factor(SchoolHoliday),
                  StateHoliday = as.factor(StateHoliday),
                  Date = as.Date(Date))]
      
      train[, month := as.factor(format(Date, "%m"))]
      test[, month := as.factor(format(Date, "%m"))]
      train[, AvgSales := Sales/Customers]
      mini <- min(train$CompetitionDistance, na.rm = T)
      maxi <- max(train$CompetitionDistance, na.rm = T)
      train[is.na(CompetitionDistance), CompetitionDistance :=  maxi]
      train[, CompNormDist := 1- (CompetitionDistance - mini)/(maxi- mini)]

      mini <- min(test$CompetitionDistance, na.rm = T)
      maxi <- max(test$CompetitionDistance, na.rm = T)
      test[is.na(CompetitionDistance), CompetitionDistance :=  maxi]
      test[, CompNormDist := 1- (CompetitionDistance - mini)/(maxi- mini)]

      train[, CompOpen := as.yearmon(paste0(CompetitionOpenSinceYear, '-', CompetitionOpenSinceMonth))]
      test[, CompOpen := as.yearmon(paste0(CompetitionOpenSinceYear, '-', CompetitionOpenSinceMonth))]
      train[, CompOpenInd := as.integer(CompOpen <= as.yearmon(Date))]
      test[, CompOpenInd := as.integer(CompOpen <= as.yearmon(Date))]
      train[ is.na(CompOpenInd), CompOpenInd := 0]
      test[ is.na(CompOpenInd), CompOpenInd := 0]
      
      train[Promo2 == 1, Promo2Since := as.Date(paste0(Promo2SinceYear, '-', Promo2SinceWeek, '-', 1), format = "%Y-%U-%u")]
      test[Promo2 == 1, Promo2Since := as.Date(paste0(Promo2SinceYear, '-', Promo2SinceWeek, '-', 1), format = "%Y-%U-%u")]
      train[, Promo2Ind := as.integer( Promo2Since <= Date & grepl(format(Date, '%b'), PromoInterval) )]
      test[, Promo2Ind := as.integer( Promo2Since <= Date & grepl(format(Date, '%b'), PromoInterval) )]
      train[is.na(Promo2Ind), Promo2Ind := 0]
      test[is.na(Promo2Ind), Promo2Ind := 0]
      
      keep <- c('Store', 'Sales', 'Customers', 'DayOfWeek', 'Date', 'Open', 'StoreType', 'Assortment',
                'month', 'AvgSales', 'CompNormDist', 'CompOpenInd', 'Promo2Ind', 'Promo')
      train <- train[,..keep]
      keep <- setdiff(keep, c('Sales', 'AvgSales', 'Customers'))
      test <- test[,..keep]
      
      return(
            list(
                  train = train,
                  test = test
            )
      )
}

