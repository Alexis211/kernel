<TeXmacs|1.99.4>

<style|generic>

<\body>
  <doc-data|<doc-title|Kaggle Project Report>|<doc-subtitle|Kernel Methods
  for Machine Learning>|<\doc-misc>
    <name|Émile Enguehard>, <name|Alex Auvolat>
  </doc-misc>>

  <subsection|Task presentation>

  The dataset is composed of a subset of the MNIST dataset, slightly altered:
  the training set is composed of 5000 labeled MNIST digits (28x28, ie 784
  features) which have been mirrored/rotated. Here are a few digits from the
  dataset:

  <center|<tabular*|<tformat|<table|<row|<cell|<small-figure|<image|192-2.png|28px|28px||>|A
  two>>|<cell|<small-figure|<image|282-9.png|28px|28px||>|A
  nine>>>|<row|<cell|<small-figure|<image|512-5.png|28px|28px||>|A
  five>>|<cell|<small-figure|<image|819-4.png|28px|28px||>|A four>>>>>>>

  As we can see, the digits are hard to recognize for a human. However the
  dataset has regularities that makes it possible to learn a good classifier
  that is adapted to this dataset.

  The test set, which is composed of 10000 digits, must be classified as
  precisely as possible. The score used for the Kaggle ranking is the
  proportion of correctly classified digits.

  <subsection|Feature extraction>

  We first convolve the images with a series of Gabor filter at six different
  orientations. The equation of a real Gabor filter which we used is:

  <\eqnarray*>
    <tformat|<table|<row|<cell|g<around*|(|x,y;\<lambda\>,\<theta\>,\<psi\>,\<sigma\>,\<gamma\>|)>>|<cell|=>|<cell|e<rsup|-<frac|x<rsub|\<theta\>><rsup|2>+\<gamma\><rsup|2>y<rsub|\<theta\>><rsup|2>|2\<sigma\><rsup|2>>>*cos<around*|(|<frac|2\<pi\>x<rsub|\<theta\>>|\<lambda\>>+\<psi\>|)>>>|<row|<cell|<matrix|<tformat|<table|<row|<cell|x<rsub|\<theta\>>>>|<row|<cell|y<rsub|\<theta\>>>>>>>>|<cell|=>|<cell|R<rsub|\<theta\>>*<matrix|<tformat|<table|<row|<cell|x>>|<row|<cell|y>>>>>>>>>
  </eqnarray*>

  The filters we used were of size <math|11\<times\>11>. The parameters were
  defined as follows:

  <\eqnarray*>
    <tformat|<table|<row|<cell|\<lambda\>>|<cell|=>|<cell|5.942>>|<row|<cell|\<theta\>>|<cell|=>|<cell|<frac|2k\<pi\>|6>,k=0\<ldots\>5>>|<row|<cell|\<psi\>>|<cell|=>|<cell|-0.556>>|<row|<cell|\<sigma\>>|<cell|=>|<cell|2.239>>|<row|<cell|\<gamma\>>|<cell|=>|<cell|1.386>>>>
  </eqnarray*>

  We then calculate for each convolved image (each corresponding to a certain
  filter orientation) the mean absolute value over patches of
  <math|4\<times\>4>. This gives a set of
  <math|<frac|28|4>\<times\><frac|28|4>\<times\>6=7\<times\>7\<times\>6=294>
  features, instead of the 784 original pixels of the image.

  We also do two kinds of normalization to help the classifier work: we
  normalize all the images independently before applying the Gabor filters,
  and we center each of the 294 features relatively to the whole dataset
  after doing the filtering and the pooling.

  <subsection|Classification>

  The classifier we use is a standard kernel <math|C>-SVM, with a RBF kernel.
  The RBF kernel is defined as follows:

  <\eqnarray*>
    <tformat|<table|<row|<cell|K<around*|(|x<rsub|>,x<rprime|'>|)>>|<cell|=>|<cell|e<rsup|-\<gamma\><around*|\<\|\|\>|x-x<rprime|'>|\<\|\|\>><rsub|2><rsup|2>>>>|<row|<cell|>|<cell|=>|<cell|<around*|\<langle\>|\<phi\><around*|(|x|)>,\<phi\><around*|(|x<rprime|'>|)>|\<rangle\>>>>>>
  </eqnarray*>

  The value of <math|\<gamma\>> we used is <math|\<gamma\>=0.7357>.

  A kernel <math|C>-SVM is defined as a solution of the following
  minimization problem:

  <\equation*>
    <tabular*|<tformat|<cwith|1|2|1|2|cell-halign|r>|<cwith|1|-1|2|2|cell-halign|l>|<cwith|1|1|3|3|cell-width|1cm>|<cwith|1|1|3|3|cell-hmode|max>|<table|<row|<cell|min<rsub|\<omega\>,\<xi\>,b>
    >|<cell|<frac|1|2><around*|\<\|\|\>|w|\<\|\|\>><rsub|2><rsup|2>+C<big|sum><rsub|i=1><rsup|m>\<xi\><rsub|i>>|<cell|>|<cell|>>|<row|<cell|subject
    to>|<cell|\<xi\><rsub|i>\<geqslant\>0>|<cell|>|<cell|i=1\<ldots\>n>>|<row|<cell|>|<cell|y<rsub|i>*<around*|(|w<rsup|T>\<phi\><around*|(|x<rsub|i>|)>+b|)>\<geqslant\>1-\<xi\><rsub|i>>|<cell|>|<cell|i=1\<ldots\>n>>>>>
  </equation*>

  Its dual is defined as follows:

  <\equation*>
    <tabular*|<tformat|<cwith|1|-1|2|2|cell-halign|l>|<cwith|1|2|1|1|cell-halign|r>|<cwith|1|1|3|3|cell-width|1cm>|<cwith|1|1|3|3|cell-hmode|max>|<table|<row|<cell|min<rsub|\<alpha\>>>|<cell|<frac|1|2><big|sum><rsub|i,j=1><rsup|n>\<alpha\><rsub|i>\<alpha\><rsub|j>y<rsub|i>y<rsub|i>K<around*|(|x<rsub|i>,x<rsub|j>|)>-\<b-1\><rsup|T>\<alpha\>>|<cell|>|<cell|>>|<row|<cell|subject
    to>|<cell|\<alpha\><rsup|T>y=0>|<cell|>|<cell|>>|<row|<cell|>|<cell|0\<leqslant\>\<alpha\><rsub|i>\<leqslant\>C>|<cell|>|<cell|i=1\<ldots\>n>>>>>
  </equation*>

  The dual is a quadratic problem which we implemented and solved using the
  <verbatim|cvxopt> library for Numpy. The classifier is the defined as
  follows, where <math|<wide|\<alpha\>|^>> is a solution of the dual:

  <\eqnarray*>
    <tformat|<table|<row|<cell|f<around*|(|x|)>>|<cell|=>|<cell|sign<around*|(|<big|sum><rsub|i=1><rsup|n><wide|\<alpha\>|^><rsub|i>y<rsub|i>K<around*|(|x,x<rsub|i>|)>+b|)>>>>>
  </eqnarray*>

  We use a one vs. one voting strategy to handle the 10 classes of the
  dataset. The parameter <math|C> that we used is <math|C=263>.

  We also implemented a <math|\<nu\>>-SVM classifier to compare the results,
  but the <math|C>-SVM was found to perform better.

  <subsection|Hyperparameter search>

  The optimal parameters for our model were found using randomized parameter
  search with 5-fold cross validation. The parameters were first set by
  intuition to be bounded by some first intervals, that we refined manually
  when good parameters were discovered. The values for the parameters were
  chosen randomly in these intervals, uniformly on a linear or logarithmic
  scale depending on the parameters.

  <subsection|Our results>

  Table 1 shows our results with the best hyperparameter combination we found
  for the <math|C>-SVM and the <math|\<nu\>>-SVM. In all cases we use a
  Gaussian RBF kernel. We see that both models are very close, but the
  <math|C>-SVM was able to perform slightly better.

  <big-table|<tabular|<tformat|<cwith|1|1|1|-1|cell-bborder|1px>|<cwith|3|3|2|-1|cell-halign|r>|<cwith|2|2|2|-1|cell-halign|r>|<table|<row|<cell|<strong|Model>>|<cell|<strong|Cross-Valid.
  error rate>>|<cell|<strong|Public score>>|<cell|<strong|Private
  score>>>|<row|<cell|Gabor filters + RBF
  <math|\<nu\>>-SVM>|<cell|2.7%>|<cell|0.9764>|<cell|0.9720>>|<row|<cell|Gabor
  filters + RBF <math|C>-SVM>|<cell|<strong|2.5%>>|<cell|<strong|0.9774>>|<cell|<strong|0.9734>>>>>>|Results
  for our method with two classifiers.>
</body>

<\initial>
  <\collection>
    <associate|page-medium|paper>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1>>
    <associate|auto-10|<tuple|1|2>>
    <associate|auto-2|<tuple|1|1>>
    <associate|auto-3|<tuple|2|1>>
    <associate|auto-4|<tuple|3|1>>
    <associate|auto-5|<tuple|4|1>>
    <associate|auto-6|<tuple|2|1>>
    <associate|auto-7|<tuple|3|2>>
    <associate|auto-8|<tuple|4|2>>
    <associate|auto-9|<tuple|5|2>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|figure>
      <tuple|normal|A two|<pageref|auto-2>>

      <tuple|normal|A nine|<pageref|auto-3>>

      <tuple|normal|A five|<pageref|auto-4>>

      <tuple|normal|A four|<pageref|auto-5>>
    </associate>
    <\associate|table>
      <tuple|normal|Results for our method with two
      classifiers.|<pageref|auto-10>>
    </associate>
    <\associate|toc>
      <with|par-left|<quote|1tab>|1<space|2spc>Task presentation
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1>>

      <with|par-left|<quote|1tab>|2<space|2spc>Feature extraction
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6>>

      <with|par-left|<quote|1tab>|3<space|2spc>Classification
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7>>

      <with|par-left|<quote|1tab>|4<space|2spc>Hyperparameter search
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-8>>

      <with|par-left|<quote|1tab>|5<space|2spc>Our results
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-9>>
    </associate>
  </collection>
</auxiliary>