<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#api-documentation">API Documentation</li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


# About The Project

This is a part of my Introduction to Data Science's assignment at university. In this part, I tried to write my own implementation of Naive Bayes Classifier from scratch!



## Built With

* [![Numpy][Numpy-shield]][Numpy-url]



# Getting Started

## Prerequisites
To use this module, your system needs to have:
* numpy
    ```sh
    pip install numpy
    ```

## Installation
You can install this module by cloning this repository into your current working directory:
```sh
git clone https://github.com/theEmperorofDaiViet/naive_bayes.git
```

# API Documentation
The **Naive_Bayes** module implements Naive Bayes algorithms. These are supervised learning methods based on applying Bayes’ theorem with strong (naive) feature independence assumptions.

## Naive_Bayes.Gaussian_Naive_Bayes

<p style="font-size: 1.17em;"><i>This model is mainly used when dealing with continuous data.</i></p>

<p style="text-align:left;">
  <pre><code>fit(X, y)</code><span style="float:right;">[<a href="https://github.com/theEmperorofDaiViet/naive_bayes/blob/master/Naive_Bayes.py#L4">source</a>]</span></pre>
</p>

<p style="margin-left: 2.5%">Fit Gaussian Naive Bayes according to X, y.</p>

<table style="width: 97.5%; margin-left: 2.5%">
  <tr>
    <th>Parameters</th>
    <td>
      <b>X: <i>np.array of shape (n_samples, n_features)</i></b><br/>
      <p style="margin-left: 2.5%">Training vectors, where <mark>n_samples</mark> is the number of samples and <mark>n_features</mark> is the number of features.</p>
      <b>y: <i>np.array of shape (n_samples)</i></b><br/>
      <p style="margin-left: 2.5%">Target values.</p><br/>
    </td>
  </tr>
  <tr>
    <th>Returns</th>
    <td>
      <b><i>None</i></b><br/>
    </td>
  </tr>
</table><br/>

<p style="text-align:left;">
  <pre><code>gaussian_density(x, mean, var)</code><span style="float:right;">[<a href="https://github.com/theEmperorofDaiViet/naive_bayes/blob/master/Naive_Bayes.py#L19">source</a>]</span></pre>
</p>

<p style="margin-left: 2.5%">Calculate the probabilit(ies) density function of Gaussian distribution for a give sample, knowing the mean(s) and the variance(s).</p>

<table style="width: 97.5%; margin-left: 2.5%">
  <tr>
    <th>Parameters</th>
    <td>
      <b>x: <i>float </i></b> or <b><i> np.array(dtype = float) of shape (n_features)</i></b><br/>
      <p style="margin-left: 2.5%">Value(s) of a feature or each feature of a certain sample.</p>
      <b>mean: <i>float </i></b> or <b><i> np.array(dtype = float) of shape (n_features)</i></b><br/>
      <p style="margin-left: 2.5%">Mean(s) of a feature or each feature.</p>
      <b>var: <i>float </i></b> or <b><i> np.array(dtype = float) of shape (n_features)</i></b><br/>
      <p style="margin-left: 2.5%">Variance(s) of a feature or each feature.</p>
    </td>
  </tr>
  <tr>
    <th>Returns</th>
    <td>
      <b>C: <i>float </i></b> or <b><i> np.array(dtype = float) of shape (n_features)</i></b><br/>
      <p style="margin-left: 2.5%">Returns the probabiliti(es) of a feature or each feature of the sample.</p>
    </td>
  </tr>
</table><br/>

<p style="text-align:left;">
  <pre><code>class_probability(x)</code><span style="float:right;">[<a href="https://github.com/theEmperorofDaiViet/naive_bayes/blob/master/Naive_Bayes.py#L24">source</a>]</span></pre>
</p>

<p style="margin-left: 2.5%">Calculate the probabilities of a given sample to belong to each class, then choose the class with maximum probability.</p>

<table style="width: 97.5%; margin-left: 2.5%">
  <tr>
    <th>Parameters</th>
    <td>
      <b>x: <i>np.array(dtype = float) of shape (n_features)</i></b><br/>
      <p style="margin-left: 2.5%">A certain sample.</p>
    </td>
  </tr>
  <tr>
    <th>Returns</th>
    <td>
      <b>C: <i>str </i></b>or <b><i>int </i></b><br/>
      <p style="margin-left: 2.5%">Returns the class which have the maximum probability of the input sample belong to it.</p>
    </td>
  </tr>
</table><br/>

<p style="text-align:left;">
  <pre><code>predict(X)</code><span style="float:right;">[<a href="https://github.com/theEmperorofDaiViet/naive_bayes/blob/master/Naive_Bayes.py#L37">source</a>]</span></pre>
</p>

<p style="margin-left: 2.5%">Perform classification on an array of test vectors X.</p>

<table style="width: 97.5%; margin-left: 2.5%">
  <tr>
    <th>Parameters</th>
    <td>
      <b>X: <i>np.array of shape (n_samples, n_features)</i></b><br/>
      <p style="margin-left: 2.5%">The input samples.</p>
    </td>
  </tr>
  <tr>
    <th>Returns</th>
    <td>
      <b>C: <i>np.array of shape (n_samples)</i></b><br/>
      <p style="margin-left: 2.5%">Predicted target values for X.</p>
    </td>
  </tr>
</table><br/>

# Usage



# Contact



<!-- MARKDOWN LINKS & IMAGES -->
[Numpy-shield]: https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white
[Numpy-url]: https://numpy.org
<style>
table, th, td {
  border: 1px solid black;
  border-collapse: collapse;
}
th {
  align: left;
  vertical-align: top;
  width: 12%
}
mark {
  background-color: gray;
  color: black;
}
</style>