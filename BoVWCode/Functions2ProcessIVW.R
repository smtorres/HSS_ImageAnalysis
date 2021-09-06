# Functions to process the BoVW in R for STM purposes
############################################## Mini functions to be passed on GetVisualSTM
# Function to get matrix ready
getDocReady <- function(vec_words){
  index <- which(vec_words!=0)
  counts <- vec_words[index]
  return(as.matrix(rbind(as.integer(index),as.integer(counts))))
}

############################################################################################

# Important: IVWMAT is the document-term matrix with an ID as the first column
####### PROCESS IVW MATRIX INTO STM DOCUMENT LIST
GetVisualSTM <- function(ivwmat,metadata,
                         outputdir,
                         ivwmatid='id', metaid = 'ID',
                         outmat = 'STM_IVWMAT_001.csv',...){
  # Check number of words
  nw = ncol(ivwmat)-1
  
  # Save the IDs
  docsids <- metadata[,metaid]
  
  # Build the vocabulary
  vocab <- colnames(ivwmat)[2:(nw+1)]
  
  # Check that all words have at least one non-zero count. 
  check <- apply(ivwmat[,-1], 2, sum)
  
  # Adjust the vocabulary and the mat accordingly
  vocab<-vocab[as.numeric(which(check!=0))]
  ivwmat <- ivwmat[,c(1,(as.numeric(which(check!=0))+1))]
  newnw <- ncol(ivwmat)
  
  # Prepare the list with documents and counts of words necessary for stm
  print('Getting the documents in shape...')
  doclist <- alply(ivwmat[,-1],1,function(x) getDocReady(x), .parallel = TRUE)
  names(doclist) <- as.character(ivwmat[,1])
  
  myobj = list(doclist=doclist, vocab=vocab)
  save(myobj, file=paste0(outputdir,"/", outmat))
  return(myobj)
}

# Function to save visual Words
saveWords=function(stmobj, numwords=15, crit='frex', dir_topics, dir_vis, content=FALSE){
  if(!content){
  labs = labelTopics(stmobj,n=numwords)
  matlab = labs[crit][[1]]
  k = nrow(matlab)
  for(i in 1:k){
    tdir = paste0(dir_topics,'/Topic',i)
    dir.create(tdir, showWarnings = FALSE)
    fls = paste0(dir_vis,'/',matlab[i,],".jpg")
    file.copy(fls, tdir)
  }
  }
  else{
    matlab= sageLabels(stmobj, n=numwords) # Rows:topics, columns: words
    marglabs = matlab['marginal']$marginal[crit][[1]]
    covslabs = matlab['cov.betas']$cov.betas
    covslabs2 = list()
    for(i in 1:length(covslabs)){
      covslabs2[[i]] <- covslabs[[i]][paste0(crit,"labels")]
    }
    names(covslabs2) <- matlab$covnames
    k = nrow(marglabs)
    for(i in 1:k){
      tdir = paste0(dir_topics,'/Topic',i)
      dir.create(tdir, showWarnings = FALSE)
      fls = paste0(dir_vis,'/', marglabs[i,],".jpg")
      file.copy(fls, tdir)
      for(j in 1:length(matlab$covnames)){
        tdir_cov = paste0(dir_topics,'/Topic',i,"/",matlab$covnames[j])
        dir.create(tdir_cov, showWarnings = FALSE)
        fls_cov = paste0(dir_vis,'/', covslabs2[[j]][[1]][i,],".jpg")
        file.copy(fls_cov, tdir_cov)
      }
    }
  }
  return(matlab)
}

# 'Create' text (visual word composition) per image
createText <- function(list, vocab){
  posword = as.numeric(list[1,])
  countword = as.numeric(list[2,])
  txt = sapply(1:length(posword), function(x) paste(rep(vocab[posword[x]],countword[x]),collapse =" " ))
  txt2 = paste(txt, collapse=" ")
  return(txt2)
}

# Choose the most representative images per topic
showTopPics <- function(stmobj, visualstm,
                        k = 10, numrepimgs = 5,
                        dir_topics, dir_imgs, isjpg=FALSE){
  txts <- as.character(unlist(lapply(stmobj[[1]], function(x) createText(x, stmobj[[2]]))))
  file_ls <- llply(1:k,function(x) {
    temp_ls <- findThoughts(visualstm, txts, topics=x,n=numrepimgs)
    index <- temp_ls$index[[1]]
    imgs_ls = names(stmobj[[1]])[index]
    if(isjpg){
      files <- paste0(dir_imgs,"/",imgs_ls,".jpg")  
    }
    else{
      files <- paste0(dir_imgs,"/",imgs_ls)  
    }
    tdir = paste0(dir_topics,'/Topic',x,'/RepImg')
    dir.create(tdir, showWarnings = FALSE)
    file.copy(files, tdir)},
    .parallel = TRUE)
  return(file_ls)
}



