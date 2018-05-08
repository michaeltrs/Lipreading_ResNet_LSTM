# Lipreading_1
In this section we discuss a model built as part of this project for visual only speech recognition. The model which follows the architecture proposed in \cite{stafylakis} is presented below. The models and data pipelines were built using Tensorflow \cite{tensorflow}. 

\section{Data}
As in \cite{stafylakis} we use the Lip Reading in the Wild (LRW) dataset \cite{lrw} which consists of videos taken from BBC news with extracts from continuous speech. Every video contains a full single word (the videos are created such that the middle of the word corresponds to the middle frame) as well as part of the previous word's ending and the beginning of the next word. For each video we have label information for what word is spoken and the duration of the word. A total of $500$ word labels exist for the dataset.\\
The facial landmark detection system presented in \cite{jdeng} was used for extracting the mouth ROI. A $(118x118)$ pixel region around the center of the mouth was extracted for each video, transformed to monochrome and saved in ".tfrecords" format. The center of the mouth is taken as the median (across all frames) of the mean mouth locations per frame (mean coordinates of all points corresponding to mouth).\\
The data are loaded and passed to the Tensorflow graph via a data pipeline. During that process data augmentation is performed by taking random crops of $(112x112)$ pixels and applying random horizontal flipping (0.5 probability of flipping) consistently across all frames in a video.
\begin{figure}
\centering % this centers the figure
\includegraphics[width = 1\hsize]{./figures/roi_extr.png} % this includes the figure and specifies that it should span 0.7 times the horizontal size of the page
\caption{Mouth ROI. Left: original frame, Middle: facial landmark detection, points are shown as red dots, Right: mouth ROI is shown in red $(118x118)$ pixel rectangle} % caption of the figure
% with a $stepsize$ $0.06$ and starting point $[1, -1]^T$
\label{fig:mouth_roi} % a label. When we refer to this label from the text, the figure number is included automatically
\end{figure}

\section{Model Architecture}
As discussed previously, the model architecture follows the one described in \cite{stafylakis}, the components of which are presented below. When dimension of tensors are provided the $"?"$ symbol is used to refer to the batch size which is not constrained by the model architecture and can vary.

\subsection{Spatiotemporal front-end}
The first model component is a 3D spatiotemporal CNN layer \cite{3dconv1} followed by batch normalization \cite{bn}, a ReLu activation \cite{relu} and a max pooling layer. Spatiotemporal convolution differs from 2D convolution in that it can extract feature dynamics in the time domain as there is explicit connectivity between the frames when passed to the 3D CNN layer. This should allow the network to capture short term spatiotemporal speech dynamics.\\
The data that are passed through the network have shape $(?, 29, 112, 112, 1)$ which corresponds to $(batch size, number of frames, height, width, number of channels)$. The 3D kernel has dimensions $(5, 7, 7, 64)$ corresponding to $(time, height, width, number of channels)$, with strides $(1, 2, 2)$. After 3D convolution the data has dimensions $(?, 29, 56, 56, 64)$ which becomes $(?, 29, 28, 28, 64)$ after a $(3x3x3)$ max pooling layer with strides $(1, 2, 2)$.

\subsection{Residual Network}
After the spatiotemporal front end the 3D features extracted are passed through a 34 layer Identity mapping version of ResNet \cite{resnet1, resnet2}. The architecture is similar to the one proposed for the ImageNet competition, however, in this case the weights and biases were initialized randomly and trained form scratch. Each frame (out of $29$ frames) is passed to ResNet at each timestep, in a manner similar to the multiple towers architecture presented in \cite{lrw}. After applying the ResNet, the data has dimensions $(?, 29, 512)$. Following ResNet, a fully connected linear layer performs dimensionality reduction leading to an output with dimensions $(?, 29, 256)$.

\subsection{Bidirectional LSTM backend}
The last component of the model is a $2$ layer bidirectional LSTM with batch normalization. This was constructed by using $2$ separate LSTM networks. The first network works with the output of the previous layer and second one works with the output of the previous layer reversed. Each LSTM's ouput is a $(?, 256)$ vectors. The outputs of the $2$ LSTMs are concatenated at every time step, leading to a $(?, 512)$ vector output. The output at the last time step is passed to a linear layer that reduces dimensions to $(?, 500)$. Finally, a Softmax layer ensures that network outputs sum to $1.0$ and define a probability distribution over outputs. It is important to note that since we have a sequence classification problem we can use only the last output for prediction. This methodology differs from the one used in \cite{stafylakis} where the authors found that applying backpropagation to every BLSTM output (using the same label for every output) yields superior results. 

\section{Training}
Training of the model was performed in a NVIDIA Titan X (12 Gb memory) GPU. Standard Stochastic Gradient Descent (SGD) was used with momentum 0.9, no dropout and BN following every layer of the network apart from the one preceding the final Softmax layer. The model was trained end-to-end with $3$ learning rate schedules. First, the model was trained for 9 epochs with learning rate $5 x 10^{-3}$ and lr decay $0.95$, then for 4 epochs with learning rate $1 x 10^{-3}$ and lr decay $0.90$ and finally for 4 epochs with learning rate $1 x 10^{-4}$ and lr decay $0.90$ per epoch.

