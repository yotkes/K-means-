#define PY_SSIZE_T_CLEAN
#include <Python.h>

static double calculate_difference(double **current_clusters, double* vector, int cluster, int d);
static int check_min_cluster(double **current_clusters, double *vector, int k, int d);
static PyObject* fit(PyObject *self, PyObject *args);
double** calc(int k, int N, int d, int MAX_ITER, int* first, double** obs);
PyMODINIT_FUNC PyInit_mykmeanssp(void);
static PyObject *Convert_Big_Array(double **array, int k, int d);

static double calculate_difference(double **current_clusters, double* vector, int cluster, int d){
    double sum = 0;
    int j;
    for (j = 0; j<d; j++){
        sum += ((vector[j] - current_clusters[cluster][j])*(vector[j] - current_clusters[cluster][j]));
    }
    return sum;
}


static int check_min_cluster(double **current_clusters, double *vector, int k, int d){
    int i;
    int cluster;
    double new_dist;
    double dist;
    i = 0;
    dist = calculate_difference(current_clusters, vector, i, d);
    for (cluster = 1; cluster<k; cluster++){
        new_dist = calculate_difference(current_clusters, vector, cluster, d);
        if (new_dist<dist){
            i = cluster;
            dist = new_dist;
        }
    }
    return i;
}

double** calc(int k, int N, int d, int MAX_ITER, int* first, double** obs){
    int i;
    int j;
    int counter;
    int centroid;
    int prev;
    int change;
    int cluster;
    double new_val;
    int *counts;
    double *p2;
    double **sums;
    int *members;
    double *p3;
    double **current_clusters;

    counts = calloc(k, sizeof(int));
    assert(counts != NULL);

    p2 = calloc(k*d, sizeof(double));
    sums = calloc(k, sizeof(double*));
    assert(p2 != NULL && sums != NULL);
    for (i = 0; i<k; i++){
        sums[i] = p2 + i*d;
    }

    members = calloc(N, sizeof(int));
    assert(members != NULL);

    p3 = calloc((k+1)*d, sizeof(double));
    current_clusters = calloc(k, sizeof(double*));
    assert(p3 != NULL && current_clusters != NULL);
    for (i = 0; i<k; i++){
        current_clusters[i] = p3 + (i+1)*d;
    }


    for(i = 0; i<N; i++){
        members[i] = k;
    }

    for(i = 0; i < k; i++) {
        for(j = 0; j < d; j++) {
            sums[i][j] = obs[first[i]][j];
            current_clusters[i][j] = obs[first[i]][j];
        }
        counts[i] = 1;
        members[first[i]] = i;
    }

    for(counter = 0; counter < MAX_ITER; counter++){
        change = 1;
        for(i = 0; i < N; i++){
            centroid = check_min_cluster(current_clusters, obs[i], k, d);
            prev = members[i];
            if(prev != centroid){
                for(j = 0; j < d; j++){
                    if (prev != k){
                        sums[prev][j] -= obs[i][j];
                    }
                    sums[centroid][j] += obs[i][j];
                }
                if (prev != k){
                    counts[prev] -= 1;
                }
                counts[centroid] += 1;
                members[i] = centroid;
            }
        }
        for(cluster = 0; cluster < k; cluster++){
            for(j = 0; j<d; j++){
                new_val = sums[cluster][j]/counts[cluster];
                if (current_clusters[cluster][j] != new_val){
                    change = 0;
                }
                current_clusters[cluster][j] = new_val;
            }
        }

        if(change){
            break;
        }
    }

   
    free(counts);
    free(sums);
    free(members);
    free(p2);
    free(p3);

    return current_clusters;
}

static PyObject* fit(PyObject *self, PyObject *args){
    int k;
    int N;
    int d;
    int MAX_ITER;
    int* first_c;
    double** obs_c;
    double** c_list;
    double* p1;
    PyObject *first_p;
    PyObject *obs_p;
    PyObject *all;
    PyObject *item;
    PyObject *py_list;
    int i;
    int j;
    
    if(!PyArg_ParseTuple(args, "O:c function getting arguments", &all)) {
        return NULL;
    }
    if (!PyList_Check(all)){
        return NULL;
    }
    k = (int)PyLong_AsLong(PyList_GetItem(all, 0));
    N = (int)PyLong_AsLong(PyList_GetItem(all, 1));
    d = (int)PyLong_AsLong(PyList_GetItem(all, 2));
    MAX_ITER = (int)PyLong_AsLong(PyList_GetItem(all, 3));
    first_p = PyList_GetItem(all, 4);
    obs_p = PyList_GetItem(all, 5);
    if (!PyList_Check(first_p)){
        return NULL;
    }
    first_c = calloc(k, sizeof(int));
    assert(first_c != NULL);
    for (i = 0; i<k; i++){
        first_c[i] = (int)PyLong_AsLong(PyList_GetItem(first_p, i));
    }

    if (!PyList_Check(obs_p)){
        return NULL;
    }

    p1 = calloc(N*d, sizeof(double));
    obs_c = calloc(N, sizeof(double*));
    assert(p1 != NULL && obs_c != NULL);
    for (i = 0; i<N; i++){
        obs_c[i] = p1 + i*d;
    }

    for (i = 0; i<N; i++){
        item = PyList_GetItem(obs_p, i);
        if (!PyList_Check(item)){
            return NULL;
        }
        for (j = 0; j<d; j++){
            obs_c[i][j] = PyFloat_AsDouble(PyList_GetItem(item, j));
        }
    }
    
    c_list = calc(k, N, d, MAX_ITER, first_c, obs_c);
    free(first_c);
    free(obs_c);
    free(p1);
    py_list = Convert_Big_Array(c_list, k, d);
    free(c_list);
    return py_list;
}

static PyObject *Convert_Big_Array(double **array, int k, int d){
    PyObject *py_list, *item;
    int i;
    int j;
    py_list = PyList_New(k*d);
    for(i = 0; i < k; i++) {
        for(j = 0; j < d; j++) {
            item = PyFloat_FromDouble(array[i][j]);
            PyList_SetItem(py_list, i*d+j, item);
        }
    }
    return py_list;
  }

static PyMethodDef capiMethods[] = {
    {"fit",                   /* the Python method name that will be used */
      (PyCFunction) fit, /* the C-function that implements the Python function and returns static PyObject*  */
      METH_VARARGS,           /* flags indicating parametersaccepted for this function */
      PyDoc_STR("kmeans calculator")}, /*  The docstring for the function */
    {NULL, NULL, 0, NULL}     /* The last entry must be all NULL as shown to act as a
                                 sentinel. Python looks for this entry to know that all
                                 of the functions for the module have been defined. */
};

/* This initiates the module using the above definitions. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp", /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,  /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    capiMethods /* the PyMethodDef array from before containing the methods of the extension */
};

PyMODINIT_FUNC PyInit_mykmeanssp(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}


