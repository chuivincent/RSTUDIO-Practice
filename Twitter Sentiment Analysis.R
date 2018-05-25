pkgs <-c('twitteR','ROAuth','httr','plyr','stringr','ggplot2','plotly')
for(p in pkgs) if(p %in% rownames(installed.packages()) == FALSE) {install.packages(p)}
for(p in pkgs) suppressPackageStartupMessages(library(p, quietly=TRUE, character.only=TRUE))

# Set API Keys
api_key <- "............................................"
api_secret <- "................................"
access_token <- ".............................."
access_token_secret <- "............................"
setup_twitter_oauth(api_key, api_secret, access_token, access_token_secret)

# Grab latest tweets
tweets_VeChain <- searchTwitter('VeChain', n=3000)
tweets_Bitcoin <- searchTwitter('Bitcoin', n=3000)
tweets_Ethereum <- searchTwitter('Ethereum', n=3000)
tweets_IOTA <- searchTwitter('IOTA', n=3000)

# Loop over tweets and extract text
feed_VeChain <- laply(tweets_VeChain, function(t) t$getText())
feed_Bitcoin <- laply(tweets_Bitcoin, function(t) t$getText())
feed_Ethereum <- laply(tweets_Ethereum, function(t) t$getText())
feed_IOTA <- laply(tweets_IOTA, function(t) t$getText())

# Read in dictionary of positive and negative workds
Positive <- scan('C:/Users/Vincent Chui/Downloads/opinion-lexicon-English/positive-words.txt',
                 what='character', comment.char=';')
Negative <- scan('C:/Users/Vincent Chui/Downloads/opinion-lexicon-English/negative-words.txt',
                 what='character', comment.char=';')
# Add a few twitter-specific negative phrases
bad_text <- c(Negative, 'wtf', 'wait', 'waiting',
              'low', 'slow')
good_text <- c(Positive, 'upgrade', 'HODL', 'hold', 'rocketship', 'rocket', 'to the moon')

score.sentiment <- function(sentences, good_text, bad_text, .progress='none')
{
  require(plyr)
  require(stringr)
  # we got a vector of sentences. plyr will handle a list
  # or a vector as an "l" for us
  # we want a simple array of scores back, so we use
  # "l" + "a" + "ply" = "laply":
  scores = laply(sentences, function(sentence, good_text, bad_text) {
    
    # clean up sentences with R's regex-driven global substitute, gsub():
    sentence = gsub('[[:punct:]]', '', sentence)
    sentence = gsub('[[:cntrl:]]', '', sentence)
    sentence = gsub('\\d+', '', sentence)
    #to remove emojis
    sentence <- iconv(sentence, 'UTF-8', 'ASCII')
    sentence = tolower(sentence)
    
    # split into words. str_split is in the stringr package
    word.list = str_split(sentence, '\\s+')
    # sometimes a list() is one level of hierarchy too much
    words = unlist(word.list)
    
    # compare our words to the dictionaries of positive & negative terms
    pos.matches = match(words, good_text)
    neg.matches = match(words, bad_text)
    
    # match() returns the position of the matched term or NA
    # we just want a TRUE/FALSE:
    pos.matches = !is.na(pos.matches)
    neg.matches = !is.na(neg.matches)
    
    # and conveniently enough, TRUE/FALSE will be treated as 1/0 by sum():
    score = sum(pos.matches) - sum(neg.matches)
    
    return(score)
  }, good_text, bad_text, .progress=.progress )
  
  scores.df = data.frame(score=scores, text=sentences)
  return(scores.df)
}

# Retreive scores and add candidate name.
VechainSentement <- score.sentiment(feed_VeChain, good_text, bad_text, .progress='text')
VechainSentement$name <- 'VeChain'
BitcoinSentement <- score.sentiment(feed_Bitcoin, good_text, bad_text, .progress='text')
BitcoinSentement$name <- 'Bitcoin'
EthereumSentement <- score.sentiment(feed_Ethereum, good_text, bad_text, .progress='text')
EthereumSentement$name <- 'Ethereum'
IOTASentement <- score.sentiment(feed_IOTA, good_text, bad_text, .progress='text')
IOTASentement$name <- 'IOTA'
# Merge into one dataframe for plotting
plotdat <- rbind(VechainSentement, BitcoinSentement, EthereumSentement, IOTASentement)
# Cut the text, just gets in the way
plotdat <- plotdat[c("name", "score")]
# Remove neutral values of 0
plotdat <- plotdat[!plotdat$score == 0, ]
# Remove anything less than -3 or greater than 3
plotdat <- plotdat[!plotdat$score > 5, ]
plotdat <- plotdat[!plotdat$score < (-5), ]

# Nice little quick plot
qplot(factor(score), data=plotdat, geom="bar", 
      fill=factor(name),
      xlab = "Sentiment Score")

# Or get funky with ggplot2 + Plotly
ep <- plotdat %>%
  ggplot(aes(x = score, fill = name)) +
  geom_bar(binwidth = 1) +
  scale_fill_manual(values = c("#0067F7","#7B00F7", "#7CF700", "#F70000")) +
  theme_classic(base_size = 12) +
  scale_x_continuous(name = "Sentiment Score") +
  scale_y_continuous(name = "Text count of tweets") +
  ggtitle("Super Tuesday Twitter Sentiment: 2016")
theme(axis.title.y = element_text(face="bold", colour="#000000", size=10),
      axis.title.x = element_text(face="bold", colour="#000000", size=8),
      axis.text.x = element_text(angle=16, vjust=0, size=8))
ggplotly(ep)