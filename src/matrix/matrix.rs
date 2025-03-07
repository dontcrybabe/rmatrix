#[derive(Debug)]
pub struct Matrix {
    cols: usize,
    rows: usize,
    data: Vec<Vec<f64>>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows],
        }
    }

    pub fn fill(&mut self, filling: f64) {
        for row in &mut self.data {
            row.fill(filling);
        }
    }
    pub fn multiply_by_scalar(&mut self, number: f64) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] *= number;
            }
        }
    }
    pub fn multiply_by_matrix(&mut self, other: &Matrix) -> Matrix {
        // Проверка, что матрицы можно перемножить
        if self.cols != other.rows {
            panic!("Matrix dimensions do not match for multiplication");
        }

        let mut result = vec![vec![0.0; other.cols]; self.rows];

        for row_idx in 0..self.rows {
            for col_idx in 0..other.cols {
                for k in 0..other.rows {
                    result[row_idx][col_idx] += self.data[row_idx][k] * other.data[k][col_idx];
                }
            }
        }
        Matrix {
            rows: other.rows,
            cols: self.cols,
            data: result,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_matrix() {
        let mut matrix = Matrix::new(1, 2);
        matrix.data = vec![vec![1.0, 2.0]];
        let mut matrix2 = Matrix::new(2, 2);
        matrix2.data = vec![vec![-3.0, 5.0], vec![4.0, -6.0]];
        let mat = matrix.multiply_by_matrix(&matrix2);
        assert_eq!(mat.data, vec![vec![5.0, -7.0]]);
    }
}
