library(readr)
library(survival)

hdSimSetup <- function(x, idVar, outcomeVar, timeVar, treatVar, 
                       form, effectRR = 1, MM = 1, nsim = 500,
                       size = nrow(x), eventRate = NULL) {
  # x = datset on which sims are based
  # idVar = name of id variable
  # outcomeVar = name of outcome variable
  # timeVar = name of the follow-up time variable
  # treatVar = name of treatment variable
  # form = RHS of formula used for outcome simulation – should look like “~ C1 + C2 + …”.  Can include anything allowed by coxph.
  # effectRR = the desired treatment effect relative risk
  # MM = multiplier of confounder effects on outcome on
  #  the log-scale
  # nsim = number of desired outcome vectors
  # size = desired size of simulated cohort studies (i.e., # of individuals)
  # eventRate = desired average event rate -- default is the event
  #  rate observed in the base dataset
  
  n <- nrow(x)
  
  sidx <- sapply(c(idVar, outcomeVar, timeVar, treatVar),
                 function(v) which(names(x) == v))
  names(x)[sidx] <- c("ID", "OUTCOME", "TIME", "TREAT")
  # print(summary(x$OUTCOME))
  # print(summary(x$TIME))
  y1 <- Surv(x$TIME, x$OUTCOME)
  y2 <- Surv(x$TIME, !x$OUTCOME)
  form1 <- as.formula(paste("y1 ~", form))
  form2 <- as.formula(paste("y2 ~", form))
  
  # estimate survival and censoring models
  smod <- coxph(form1, x = TRUE, data = x)
  # print(summary(smod))
  fit <- survfit(smod) 
  s0 <- fit$surv      # survival curve for average patient
  ts <- fit$time
  nts <- length(ts)
  cmod <- coxph(form2, data = x)
  # print(summary(cmod))
  fit <- survfit(cmod) 
  c0 <- fit$surv      # censoring curve for average patient
  
  # find event rate in base cohort (if everyone was followed to end of study)
  Xb <- as.vector(smod$x %*% coef(smod))
  mx <- colMeans(smod$x)
  xb0 <- mx %*% coef(smod)
  s0end <- min(s0)
  if(is.null(eventRate)) eventRate <- 1-mean(s0end^exp(Xb - xb0))
  print(eventRate)
  
  # find delta value needed to get approximate desired event rate under new parameters
  bnew <- replace(MM*coef(smod), names(coef(smod)) == "TREAT", log(effectRR))
  Xbnew <- as.vector(smod$x %*% bnew)
  sXend <- s0end^(exp(Xb - xb0))
  # print(sXend)
  # print(eventRate)
  fn <- function(d) mean(sXend^d) - (1 - eventRate)
  delta <- uniroot(fn, lower = 0, upper = 10000)$root
  
  # setup n X nts matrix of individual survival and censoring curves under new parameters
  Sx <- matrix(unlist(lapply(s0, function(s) s^(delta*exp(Xbnew - xb0)))), nrow = n)
  Xbnew <- as.vector(smod$x %*% coef(cmod))
  xb0 <- mx %*% coef(cmod)
  Cx <- matrix(unlist(lapply(c0, function(s) s^(delta*exp(Xbnew - xb0)))), nrow = n)
  
  #### sample and simulate
  ids <- tnew <- ynew <- data.frame(matrix(nrow = size, ncol = nsim))
  for(sim in 1:nsim) {
    idxs <- sample(n, size, replace = TRUE)
    ids[,sim] <- x$ID[idxs]
    
    # event time
    u <- runif(size, 0, 1)
    w <- apply(Sx[idxs,] < u, 1, function(x) which(x)[1]) # the first time survival drops below u
    stime <- ts[w]
    w <- Sx[idxs,nts] > u     # for any individuals with survival that never drops below u, 
    stime[w] <- max(ts) + 1  # replace with arbitrary time beyond last observed event/censoring time   
    
    # censoring time
    u <- runif(size, 0, 1)
    w <- apply(Cx[idxs,] < u, 1, function(x) which(x)[1]) # the first time censor-free survival drops below u
    ctime <- ts[w]
    w <- Cx[idxs,nts] > u     # for any individuals with censor-free survival that never drops below u, 
    ctime[w] <- max(ts)    # replace with hard censor time at last observed event/censoring time   
    
    # put it together
    tnew[,sim] <- pmin(stime, ctime)
    names(tnew) <- paste("TIME", 1:nsim, sep = "")
    ynew[,sim] <- stime == tnew[,sim]
    names(ynew) <- paste("EVENT", 1:nsim, sep = "")
  }
  
  # names(ids) <- paste("ID", 1:nsim, sep = "")
  # names(tnew) <- paste("TIME", 1:nsim, sep = "")
  names(ynew) <- paste("EVENT", 1:nsim, sep = "")

  # transform ynew to vector, which help construct data.frame later
  # ynew <- as.vector(ynew)
  
  return(ynew)
  # write.csv(data.frame(ids, ynew, tnew), "test.csv", row.names = FALSE)
  # return(data.frame(ids, ynew, tnew))
}

args <- commandArgs(trailingOnly = TRUE)

# Example: assuming your script requires two arguments
if (length(args) < 3) {
  stop("Insufficient arguments provided")
}

input_data_path <- args[1]
gen_sample_size <- as.integer(args[2])
output_data_path <- args[3]

mimic <- read_csv(input_data_path)

variables <- names(mimic)
event_D_vars <- variables[grep("^event_D_", variables)]
mimic$trt <- rbinom(nrow(mimic), 1, 0.5)

rhs <- "GENDER + factor(AGE_GROUP) + factor(MERGED_ETHNICITY)"

# enumerate event_D_vars
# results <- list()
print(length(event_D_vars))
for (event_D_var in event_D_vars) {
  print(event_D_var)
  start_time <- Sys.time()
  # construct timeVar by changing event_D_* to TimeToEvent_D_*
  timeVar <- gsub("event_D_", "TimeToEvent_D_", event_D_var)
  # Attempt to simulate data with error handling
  ynew <- tryCatch({
    # Attempt the simulation
    hdSimSetup(mimic, idVar = "SUBJECT_ID", outcomeVar = event_D_var, 
               timeVar = timeVar, treatVar = "trt", 
               form = rhs, nsim = 1, size = gen_sample_size)
  }, error = function(e) {
    # If an error occurs, print it and return NULL
    print(paste("Error processing", event_D_var, ":", e$message))
    return(NULL)
  })
  # If ynew is not NULL, collect the result
  if (!is.null(ynew)) {
    # results[[event_D_var]] <- ynew
    write.csv(data.frame(ynew), paste(output_data_path, "/", event_D_var, ".csv", sep = ""), row.names = FALSE)
  }
}