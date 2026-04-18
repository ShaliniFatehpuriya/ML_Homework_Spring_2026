#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>
#include <random>
using namespace std;
/*--------------------------Copied from assignment 2-------------------- */
//reading the data from bytes
int readData(ifstream &file){
    unsigned char bytes[4];
    //reading the 32 bit integers (file is in bytes)
    file.read((char*)bytes, 4);
    //x<<n == x*2^n (left shift operator)
    return (int(bytes[0]) << 24) |
           (int(bytes[1]) << 16) |
           (int(bytes[2]) << 8)  |
           (int(bytes[3]));
}
//filling the image data fro the file
void readDataImage(string fileName, vector<vector<unsigned char>> &imageData){
    ifstream file(fileName, ios::binary);
    if(!file){
        cout<<"Error Reading file";
        return;
    }

    int magicNumber = readData(file);
    int numOfImage = readData(file);
    int rows=readData(file);
    int cols=readData(file);

    //memory allocation for array
    imageData.resize(numOfImage, vector<unsigned char>(rows * cols));

    for (int i = 0; i < numOfImage; i++) {
        file.read((char*)imageData[i].data(), rows * cols);
    }
    file.close();
}
void readDataLabels(string filename, vector<unsigned char> &labels) {
    ifstream file(filename, ios::binary);
    if (!file) {
        cout << "Cannot open file!" << endl;
        return;
    }
    int magic = readData(file);
    int numLabels = readData(file);
    labels.resize(numLabels);
    file.read((char*)labels.data(), numLabels);
    file.close();
}
int getBinNum(unsigned char pixel){
    return pixel>127?1:0;
}
//EM ALgorithm Probability 
struct EM {
    int numClusters = 10;        // number of clusters 10 for each image
    int numPixels = 784;         // 28X28 pixels 
    int numImages;               // number of training images
    
    vector<double> pi;           // pi[k] = probability of choosing cluster k
    vector<vector<double>> mu;   // mu[k][j] = probability pixel j is 1 in cluster k
};
//Converting the image value to binary data (using get bin method from assignment 2)
vector<vector<int>> binarizeImages(const vector<vector<unsigned char>>& images) {
    vector<vector<int>> binaryImages(images.size(), vector<int>(images[0].size()));
    for (int i = 0; i < images.size(); i++) {
        for (int j = 0; j < images[i].size(); j++) {
            binaryImages[i][j] = getBinNum(images[i][j]);
        }
    }
    return binaryImages;
}
// We start with random guesses for mu (pixel probabilities)
void initializeEM(EM& model, int numImages) {
    model.numImages = numImages;
    // Initialize pi - each cluster equally likely at start //every element is 1/10 here
    model.pi.assign(model.numClusters, 1.0 / model.numClusters);
    //a vector of 10 rows and each row has a prob of 784 pirxels
    model.mu.resize(model.numClusters, vector<double>(model.numPixels));
    // Initialize mu - random probabilities between 0.2 and 0.8 (stable logs)
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.2, 0.8);
    for (int k = 0; k < model.numClusters; k++) {
        for (int j = 0; j < model.numPixels; j++) {
            model.mu[k][j] = dis(gen);
        }
    }
}
//each image we get probability it belongs to each cluster
// r_ik = P(cluster=k | image_i) = (pi_k * P(image_i|cluster=k)) / sum over k
vector<vector<double>> expectationStep(const vector<vector<int>>& images, const EM& model) {
    int N = images.size();
    int K = model.numClusters;
    vector<vector<double>> responsibilities(N, vector<double>(K));    
    
    for (int i = 0; i < N; i++) {
        vector<double> logProbs(K);
        double maxLogProb = -1e100;
        
        // Calculate log probabilities
        for (int k = 0; k < K; k++) {
            double logProb = log(model.pi[k]);
            for (int j = 0; j < model.numPixels; j++) {
                double mu_kj = max(min(model.mu[k][j], 1.0 - 1e-10), 1e-10); // Clamp to avoid 0/1
                if (images[i][j] == 1) {
                    logProb += log(mu_kj);
                } else {
                    logProb += log(1.0 - mu_kj);
                }
            }
            logProbs[k] = logProb;
            if (logProb > maxLogProb) maxLogProb = logProb;
        }
        // Convert to probabilities using log-sum-exp trick
        double total = 0.0;
        vector<double> probs(K);
        for (int k = 0; k < K; k++) {
            probs[k] = exp(logProbs[k] - maxLogProb);
            total += probs[k];
        }
        // Normalize
        if (total > 0) {
            for (int k = 0; k < K; k++) {
                responsibilities[i][k] = probs[k] / total;
            }
        } else {
            // If total is 0, assign equal probabilities
            for (int k = 0; k < K; k++) {
                responsibilities[i][k] = 1.0 / K;
            }
        }
    }
    return responsibilities;
}
// uppdate mu and pi based on weighted averages
//  mu[k][j] = sum[i]( rres[i][j] * X[i][j]) / (sum_i r_ik) anddd pi_k = (sum_i r_ik) / N
void maximizationStep(const vector<vector<int>>& images, const vector<vector<double>>& responsibilities, EM& model) {
    int N = images.size();
    int K = model.numClusters;
    int P = model.numPixels;
    // Calculate Nk = total responsibility for each cluster
    vector<double> Nk(K, 0.0);
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < K; k++) {
            Nk[k] += responsibilities[i][k];
        }
    }
    // Update pi (cluster probabilities) total respo of that cluster/number of imgaes
    for (int k = 0; k < K; k++) {
        model.pi[k] = Nk[k] / N;
    }
    // Update mu (pixel probabilities for each cluster) (resp of each cluster for image i * image pixel 0 or 1)/responsibiluty [i][k] == its for that specific pixel
    for (int k = 0; k < K; k++) {
        for (int j = 0; j < P; j++) {
            double weightedSum = 0.0;
            for (int i = 0; i < N; i++) {
                weightedSum += responsibilities[i][k] * images[i][j];
            }
            // Avoid division by zero
            model.mu[k][j] = (Nk[k] > 0) ? weightedSum / Nk[k] : 0.5;
        }
    }
}
// csalculate log-likelihood to check convergence it measures how well our model fits the data
double computeLogLikelihood(const vector<vector<int>>& images, const EM& model) {
    double logLikelihood = 0.0;
    int N = images.size();
    int K = model.numClusters;
    for (int i = 0; i < N; i++) {
        vector<double> probs(K);
        double maxLogProb = -1e100;
        // compute all log probabilities and find max
        for (int k = 0; k < K; k++) {
            double logProb = log(model.pi[k]);
            for (int j = 0; j < model.numPixels; j++) {
                if (images[i][j] == 1) {
                    logProb += log(max(model.mu[k][j], 1e-10));  // Avoid log(0)
                } else {
                    logProb += log(max(1.0 - model.mu[k][j], 1e-10));  // Avoid log(0)
                }
            }
            probs[k] = logProb;
            if (logProb > maxLogProb) maxLogProb = logProb;
        }
        // Use log-sum-exp trick for numerical stability
        double sum = 0.0;
        for (int k = 0; k < K; k++) {
            sum += exp(probs[k] - maxLogProb);
        }
        logLikelihood += maxLogProb + log(sum);
    }
    
    return logLikelihood;
}
//assssign each image to its most likely cluster
vector<int> assignClusters(const vector<vector<double>>& responsibilities) {
    vector<int> assignments(responsibilities.size());
    
    for (int i = 0; i < responsibilities.size(); i++) {
        int bestCluster = 0;
        double bestProb = responsibilities[i][0];
        
        for (int k = 1; k < responsibilities[i].size(); k++) {
            if (responsibilities[i][k] > bestProb) {
                bestProb = responsibilities[i][k];
                bestCluster = k;
            }
        }
        assignments[i] = bestCluster;
    }
    
    return assignments;
}
//map cluster numbers to actual digits 
vector<int> mapClustersToDigits(const vector<int>& assignments, const vector<unsigned char>& trueLabels) {
    int K = 10;
    //true lables are the lables we got initially
    vector<vector<int>> clusterLabels(K);
    // Collect all true labels for each cluster
    for (int i = 0; i < assignments.size(); i++) {
        clusterLabels[assignments[i]].push_back(trueLabels[i]);
    }
    // Find most common digit in each cluster
    vector<int> clusterToDigit(K);
    for (int k = 0; k < K; k++) {
        if (clusterLabels[k].empty()) {
            clusterToDigit[k] = k;  // Default if empty
            continue;
        }
        
        vector<int> digitCount(10, 0);
        for (int label : clusterLabels[k]) {
            digitCount[label]++;
        }
        
        int maxDigit = 0;
        int maxCount = digitCount[0];
        for (int d = 1; d < 10; d++) {
            if (digitCount[d] > maxCount) {
                maxCount = digitCount[d];
                maxDigit = d;
            }
        }
        clusterToDigit[k] = maxDigit;
    }
    
    return clusterToDigit;
}
// prirint confusion matrix 
void printResults(const vector<int>& assignments,const vector<int>& clusterToDigit,const vector<unsigned char>& trueLabels) {    
    // For each digit 0-9
    for (int digit = 0; digit < 10; digit++) {
        int truePos = 0, falseNeg = 0, falsePos = 0, trueNeg = 0;
        // Calculate confusion matrix values
        for (int i = 0; i < assignments.size(); i++) {
            int predictedDigit = clusterToDigit[assignments[i]];
            int trueDigit = trueLabels[i];
            if (trueDigit == digit) {
                if (predictedDigit == digit) truePos++;
                else falseNeg++;
            } else {
                if (predictedDigit == digit) falsePos++;
                else trueNeg++;
            }
        }
        
        // Print confusion matrix for this digit
        cout << "Confusion Matrix " << digit << ":" << endl;
        cout << "\t\tPredicted "<<digit << "\tPredicted not "<<digit <<endl;
        cout << "is number " << digit << "\t       "<<truePos <<"\t"<<falseNeg<< endl;
        cout << "isnt number " << digit << "\t       "<<falsePos <<"\t"<<trueNeg<< endl;
        // Calculate and print sensitivity and specificity
        double sensitivity = (truePos + falseNeg > 0) ? (double)truePos / (truePos + falseNeg) : 0;
        double specificity = (falsePos + trueNeg > 0) ? (double)trueNeg / (falsePos + trueNeg) : 0;
        
        cout << "Sensitivity (true positive rate): " << sensitivity << endl;
        cout << "Specificity (true negative rate): " << specificity << endl;
        cout << endl;
    }
}
// Shows what the model thinks each digit looks like
void printImagination(const EM& model, const vector<int>& clusterToDigit) {    
    for(int digit = 0; digit < 10; digit++) {
        // Find which cluster represents this digit
        int cluster = -1;
        for (int k = 0; k < 10; k++) {
            if (clusterToDigit[k] == digit) {
                cluster = k;
                break;
            }
        }
        if (cluster == -1) {
            cout << "Digit " << digit << ": No cluster found" << endl;
            continue;
        }
        cout << "Class :" << digit << endl;
        // pixel prinbts
        for (int row = 0; row < 28; row++) {
            for (int col = 0; col < 28; col++) {
                int pixelIdx = row * 28 + col;
                // If probability > 0.5, print 1, else 0
                if (model.mu[cluster][pixelIdx] > 0.5) {
                    cout << "1";
                } else {
                    cout << "0";
                }
            }
            cout << endl;
        }
        cout << endl;
    }
}

int main(int argc, char const *argv[]){
    EM model; //getting a model of type EM
    // Load the data
    vector<vector<unsigned char>> trainImage;
    vector<unsigned char> trainLabel;
    readDataImage("./Data/train-images.idx3-ubyte___", trainImage);
    readDataLabels("./Data/train-labels.idx1-ubyte___", trainLabel);
    // each image is now either 1 or 0 fir each pixel
    vector<vector<int>> binaryImages = binarizeImages(trainImage); 
    binaryImages.resize(1000);
    trainLabel.resize(1000);
    //Initialising the EM model (initial probablities)   
    initializeEM(model, binaryImages.size());
    // Running the EM algorithm  
    int maxIterations = 10;
    double epsilon = 1e-6;   // Convergence
    double prevLogLikelihood = -1e100;
    
    for (int iter = 0; iter < maxIterations; iter++) {
        // E-step
        vector<vector<double>> responsibilities = expectationStep(binaryImages, model);
        // M-step
        maximizationStep(binaryImages, responsibilities, model);
        // Check convergence
        double currentLogLikelihood = computeLogLikelihood(binaryImages, model);
        double diff = fabs(currentLogLikelihood - prevLogLikelihood);        
        if (diff < epsilon) {
            break;
        }
        prevLogLikelihood = currentLogLikelihood;
        cout << endl;
    }
    // assigning each image to its most likely cluster
    vector<vector<double>> finalResponsibilities = expectationStep(binaryImages, model);
    vector<int> assignments = assignClusters(finalResponsibilities);
    //map clusters to actual digits
    vector<int> clusterToDigit = mapClustersToDigits(assignments, trainLabel);
    
    //Print results
    printImagination(model, clusterToDigit);
    printResults(assignments, clusterToDigit, trainLabel);    
    // Calculate overall error rate
    int correct = 0;
    for (int i = 0; i < assignments.size(); i++) {
        if (clusterToDigit[assignments[i]] == trainLabel[i]) {
            correct++;
        }
    }
    double errorRate = 1.0 - (double)correct / assignments.size();
    cout<<"Total iteration to converge: "<<maxIterations<<endl;
    cout<<"Total error rate: "<<errorRate;
    
    return 0;
}
