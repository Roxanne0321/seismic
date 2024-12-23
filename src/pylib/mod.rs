use crate::{InvertedIndex, SparseDataset};
use half::f16;
use numpy::PyReadonlyArrayDyn;
use numpy::{PyArray2, IntoPyArray};
use ndarray::Array2;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::fs;
use crate::inverted_index::{BlockingStrategy, Configuration, PruningStrategy, SummarizationStrategy};

#[pyclass]
pub struct PySeismicIndex {
    inverted_index: InvertedIndex<f16>,
    queries: Vec<(Vec<usize>, Vec<f32>)>,
    results: Vec<Vec<(f32, usize)>>,
}


#[pymethods]
impl PySeismicIndex {
    #[staticmethod]
    pub fn load(index_path: &str) -> PyResult<PySeismicIndex> {
        let serialized: Vec<u8> = fs::read(index_path).unwrap();
        let inverted_index = bincode::deserialize::<InvertedIndex<f16>>(&serialized).unwrap();
        Ok(PySeismicIndex { 
            inverted_index,
            queries: vec![],
            results: vec![], 
        })
    }

    pub fn save(&self, path: &str) {
        let serialized = bincode::serialize(&self.inverted_index).unwrap();
        let path = path.to_string() + ".index.seismic";
        println!("Saving ... {}", path);
        let r = fs::write(path, serialized);
        println!("{:?}", r);
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn build(input_file: &str,
                 n_postings: usize,
                 centroid_fraction: f32,
                 truncated_kmeans_training: bool,
                 truncation_size: usize,
                 min_cluster_size: usize,
                 summary_energy: f32) -> PyResult<PySeismicIndex> {
        let dataset = SparseDataset::<f32>::read_bin_file(input_file)
            .unwrap()
            .quantize_f16();

        let config = Configuration::default()
            .pruning_strategy(PruningStrategy::GlobalThreshold {
                n_postings,
                max_fraction: 1.5,
            })
            .blocking_strategy(BlockingStrategy::RandomKmeans {
                centroid_fraction,
                truncated_kmeans_training,
                truncation_size,
                min_cluster_size,
            })
            .summarization_strategy(SummarizationStrategy::EnergyPerserving {
                summary_energy,
            });
        println!("\nBuilding the index...");
        println!("{:?}", config);

        let inverted_index = InvertedIndex::build(dataset, config);
        Ok(PySeismicIndex { 
            inverted_index,
            queries: vec![],
            results: vec![], 
        })
    }

    pub fn search<'py>(
        &self,
        query_components: PyReadonlyArrayDyn<'py, i32>,
        query_values: PyReadonlyArrayDyn<'py, f32>,
        k: usize,
        query_cut: usize,
        heap_factor: f32,
    ) -> Vec<(f32, usize)> {
        self.inverted_index.search(
            &query_components
                .to_vec()
                .unwrap()
                .iter()
                .map(|x| *x as u16)
                .collect::<Vec<_>>(),
            &query_values.to_vec().unwrap(),
            k,
            query_cut,
            heap_factor,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn batch_search<'py>(
        &self,
        k: usize,
        query_cut: usize,
        heap_factor: f32, 
    ) -> PyResult<()> {
        // 设置 rayon 的线程池
        rayon::ThreadPoolBuilder::new().build_global().unwrap();

        batch_queries = self.queries.into_iter().collect::<SparseDataset<f32>>().into();
    
        // 并行处理查询
        self.results = batch_queries
            .par_iter()
            .map(|query| {
                self.inverted_index.search(
                    &query.0,
                    &query.1,
                    k,
                    query_cut,
                    heap_factor,
                )
            })
            .collect::<Vec<_>>();

        Ok(())
    }

    #[allow(clippy::type_complexity)]
    pub fn process_queries<'py>(&self, py: Python<'py>, query_data: &'py PyAny) -> PyResult<()> {
        // 尝试将 query_data 解析为 PyList
        let rows: &PyList = query_data.downcast::<PyList>()?;
    
        // 准备 Rust 的 Vec<(Vec<usize>, Vec<f32>)> 类型
        self.queries = Vec::with_capacity(rows.len());

        for row in rows.iter() {
            let tuple: &PyTuple = row.downcast::<PyTuple>()?;
        
            // 获取 indices 和 values
            let indices_pylist: &PyList = tuple.get_item(0).downcast::<PyList>()?;
            let values_pylist: &PyList = tuple.get_item(1).downcast::<PyList>()?;
        
            // 提取为 Vec<usize> 和 Vec<f32>
            let indices: Vec<usize> = indices_pylist.extract()?;
            let values: Vec<f32> = values_pylist.extract()?;
        
            // 追加到 vec 中
            self.queries.push((indices, values));
        }

        Ok(())
    }

    pub fn get_results<'py> (&self, py: Python<'py>, k: usize) -> Vec<Vec<usize>>{
        // 提取 usize 项
        let num_queries = self.results.len();
        let mut indices_data = Vec::with_capacity(num_queries);

        for result_row in self.results.iter() {
            let mut indices_row = Vec::with_capacity(k);
        
            for &(_score, idx) in result_row.iter() {
                indices_row.push(idx);
            }
        
            indices_data.push(indices_row);
        }
        indices_data
    }
}
