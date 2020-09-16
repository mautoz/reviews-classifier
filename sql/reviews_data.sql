CREATE TABLE IF NOT EXISTS reviews_data (
    id                  SERIAL NOT NULL PRIMARY KEY,
    user_name           text,

    -- Google or Apple
    review_source       text,
    review_app          text,
    review_date         TIMESTAMP not NULL,
    review              TEXT not NULL,
    -- up_vote se é de acessibilidade
    -- down_vote caso contrário
    up_vote             INTEGER DEFAULT 0, 
    down_vote           INTEGER DEFAULT 0,

);

CREATE TABLE IF NOT EXISTS a11y_words (
    -- Accessibility (a11y)

    id                  SERIAL NOT NULL PRIMARY KEY,
    reviews_data_id     INTEGER not NULL REFERENCES reviews_data(id),
    word                text not NULL,    
)