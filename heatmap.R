# Example matrix
setwd('/Users/alejandroadriaquelozano/Documents/Systems Biology/Project 2/Results CI opt/')
X= read.csv('X_CI_optfast.csv')
X= as.matrix(X)
# Example classes
classes <- c("anger", "disgust", "happiness", "sadness", "fear")

# Assign random classes to each row in the matrix
Y <- read.csv('y_CI.csv')
Y<- Y$y_emotions
# Create a color palette for the classes
class_colors <- c("anger" = "red", "disgust" = "green", "sadness" = "blue", "happiness" = "yellow", "fear" = "purple")

# Convert the classes to corresponding colors
row_colors <- class_colors[Y]


# Create the heatmap with color-coded y-axis
heatmap(X, col = hcl.colors(50), RowSideColors = row_colors, Rowv = NA, Colv = NA, xlab = "Channels", ylab = "",main='CI opt fast region',las=2)

par(mar = c(5, 6, 4, 2) + 0.1)  # Adjust the margin to make room for the rotated label
text(0.8, 0.8, "Samples", srt = 90, adj = c(1, 0.5), xpd = TRUE)


legend(x=0.8,y=0.4,fill = rev(unique(row_colors)), legend = rev(unique(Y)), title = "Classes", cex = 0.5)

