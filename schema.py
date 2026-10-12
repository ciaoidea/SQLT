"""Create the SQLT native VECTOR tables without loading a model."""
from database import connect

def initialize():
    statements=[
    '''CREATE TABLE IF NOT EXISTS moe_model(id INT PRIMARY KEY,repo TEXT,source_commit CHAR(40),config JSON,generation_config JSON,tokenizer_config JSON,tokenizer_structure JSON,model_card LONGTEXT,status VARCHAR(20),revision BIGINT)''',
    '''CREATE TABLE IF NOT EXISTS moe_parameters(id INT AUTO_INCREMENT PRIMARY KEY,name VARCHAR(200) UNIQUE,layer_no INT,expert_no INT,shape JSON,dimensions INT,row_count INT,sha256 CHAR(64),role VARCHAR(30),UNIQUE(name,expert_no))''',
    '''CREATE TABLE IF NOT EXISTS moe_scalars(parameter_id INT,unit_no INT,coefficient DOUBLE,PRIMARY KEY(parameter_id,unit_no),FOREIGN KEY(parameter_id) REFERENCES moe_parameters(id))''',
    '''CREATE TABLE IF NOT EXISTS moe_tokens(id INT PRIMARY KEY,piece TEXT,added BOOLEAN,in_vocabulary BOOLEAN,properties JSON)''',
    '''CREATE TABLE IF NOT EXISTS moe_merges(rank_no INT PRIMARY KEY,first_piece TEXT,second_piece TEXT)''',
    '''CREATE TABLE IF NOT EXISTS moe_layers(layer_no INT PRIMARY KEY,expert_count INT,active_experts INT,router_parameter INT,FOREIGN KEY(router_parameter) REFERENCES moe_parameters(id))''',
    '''CREATE TABLE IF NOT EXISTS moe_experts(layer_no INT,expert_no INT,input_parameter INT,output_parameter INT,PRIMARY KEY(layer_no,expert_no),FOREIGN KEY(input_parameter) REFERENCES moe_parameters(id),FOREIGN KEY(output_parameter) REFERENCES moe_parameters(id))''',
    '''CREATE TABLE IF NOT EXISTS moe_events(id BIGINT AUTO_INCREMENT PRIMARY KEY,kind VARCHAR(40),metrics JSON,created TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''']
    with connect() as c,c.cursor() as cur:
        for sql in statements:cur.execute(sql)
        for dimension in (512,1536):
            cur.execute(f'''CREATE TABLE IF NOT EXISTS moe_vectors_{dimension}(parameter_id INT,unit_no INT,coefficients VECTOR({dimension}) NOT NULL,PRIMARY KEY(parameter_id,unit_no),FOREIGN KEY(parameter_id) REFERENCES moe_parameters(id))''')
        c.commit()
        initialize_history(c)


def initialize_history(connection):
    # DDL is deliberately outside the weight-update transaction.
    with connection.cursor() as cur:
        cur.execute('''CREATE TABLE IF NOT EXISTS moe_training_runs (
            id BIGINT AUTO_INCREMENT PRIMARY KEY, base_revision BIGINT NOT NULL,
            saved_revision BIGINT NOT NULL, metrics JSON NOT NULL,
            created TIMESTAMP DEFAULT CURRENT_TIMESTAMP) ENGINE=InnoDB''')
        cur.execute('''CREATE TABLE IF NOT EXISTS moe_training_parameters (
            run_id BIGINT NOT NULL, parameter_id INT NOT NULL, sha256 CHAR(64) NOT NULL,
            PRIMARY KEY(run_id,parameter_id)) ENGINE=InnoDB''')
        for dim in (0, 512, 1536):
            table = f'moe_training_vectors_{dim}' if dim else 'moe_training_scalars'
            field = f'coefficients VECTOR({dim})' if dim else 'coefficient DOUBLE'
            cur.execute(f'''CREATE TABLE IF NOT EXISTS {table} (
                run_id BIGINT NOT NULL, parameter_id INT NOT NULL, unit_no INT NOT NULL,
                {field} NOT NULL, PRIMARY KEY(run_id,parameter_id,unit_no)) ENGINE=InnoDB''')
    connection.commit()
