require(mxnet)

AutoEncoder <-
  setRefClass(
    "AutoEncoder", fields = c(
      "data", "N", "dims", "stacks", "pt_dropout",
      "ft_dropout", "input_act",
      "internal_act", "output_act"
    )
  )

make_stack <-
  function(ctx, istack, data, num_input, num_hidden, sparseness_penalty = NULL,
           idropout = NULL, odropout = NULL,
           encoder_act = 'relu', decoder_act = 'relu') {
    x <- data
    if (!is.null(idropout)) {
      x <- mx.symbol.Dropout(data = x, p = idropout)
    }
    
    x = mx.symbol.FullyConnected(
      name = paste0('encoder_', istack), data = x, num.hidden = num_hidden
    )
    
    if (!is.null(encoder_act)) {
      x = mx.symbol.Activation(data = x, act.type = encoder_act)
      if (encoder_act == 'sigmoid' &&
          !is.null(sparseness_penalty)) {
        x <- mx.symbol.IdentityAttachKLSparseReg(
          data = x, name = paste0('sparse_encoder_', istack),
          penalty = sparseness_penalty
        )
        
      }
    }
    
    if (!is.null(odropout)) {
      x <- mx.symbol.Dropout(data = x, p = odropout)
    }
    
    x <- mx.symbol.FullyConnected(
      name = paste0('decoder_', istack),
      data = x, num.hidden = num_input
    )
    
    if (decoder_act == 'softmax') {
      x = mx.symbol.Softmax(
        data = x, label = data, prob.label = TRUE, act.type = decoder_act
      )
    } else if (!is.null(decoder_act)) {
      x = mx.symbol.Activation(data = x, act.type = decoder_act)
      if (decoder_act == 'sigmoid' &&
          !is.null(sparseness_penalty)) {
        x = mx.symbol.IdentityAttachKLSparseReg(
          data = x, name = paste0('sparse_decoder_', istack), penalty = sparseness_penalty
        )
      }
      x = mx.symbol.LinearRegressionOutput(data = x, label = data)
    } else {
      x = mx.symbol.LinearRegressionOutput(data = x, label = data)
    }
    
    args <- list()
    args[[paste0('encoder_', istack, '_weight')]] <-
      mxnet:::mx.nd.internal.empty(c(num_input, num_hidden), ctx)
    args[[paste0('encoder_', istack, '_bias')]] <-
      mxnet:::mx.nd.internal.empty(num_hidden, ctx)
    args[[paste0('decoder_', istack, '_weight')]] <-
      mxnet:::mx.nd.internal.empty(c(num_hidden, num_input), ctx)
    args[[paste0('decoder_', istack, '_bias')]] <-
      mxnet:::mx.nd.internal.empty(num_input, ctx)
    
    args_grad <- list()
    args_grad[[paste0('encoder_', istack, '_weight')]] <-
      mxnet:::mx.nd.internal.empty(c(num_input, num_hidden), ctx)
    args_grad[[paste0('encoder_', istack, '_bias')]] <-
      mxnet:::mx.nd.internal.empty(num_hidden, ctx)
    args_grad[[paste0('decoder_', istack, '_weight')]] <-
      mxnet:::mx.nd.internal.empty(c(num_hidden, num_input), ctx)
    args_grad[[paste0('decoder_', istack, '_bias')]] <-
      mxnet:::mx.nd.internal.empty(num_input, ctx)
    
    args_mult <- list()
    args_mult[[paste0('encoder_', istack, '_weight')]] <- 1.0
    args_mult[[paste0('encoder_', istack, '_bias')]] <- 2.0
    args_mult[[paste0('decoder_', istack, '_weight')]] <- 1.0
    args_mult[[paste0('decoder_', istack, '_bias')]] <- 2.0
    
    auxs <- list()
    
    if (encoder_act == 'sigmoid' && !is.null(sparseness_penalty)) {
      auxs[[paste0('sparse_encoder_', istack, '_moving_avg')]] = mx.nd.ones(num_hidden, ctx) * 0.5
    }
    
    if (decoder_act == 'sigmoid' && !is.null(sparseness_penalty)) {
      auxs[[paste0('sparse_decoder_', istack, '_moving_avg')]] = mx.nd.ones(num_input, ctx) * 0.5
    }
    
    init = mx.init.uniform(0.07)
    
    for (i in 1:length(names(args))) {
      init(names(args)[i], args[[i]], ctx)
    }
    
    return(
      list(
        'x' = x, 'args' = args, 'args_grad' = args_grad,
        'args_mult' = args_mult, 'auxs' = auxs
      )
    )
  }

AE_setup <-
  function(ctx, dims, sparseness_penalty = NULL, pt_dropout = NULL, ft_dropout = NULL,
           input_act = NULL, internal_act = 'relu', output_act = NULL) {
    data = mx.symbol.Variable('data')
    ae_model <- AutoEncoder$new('data' = data)
    ae_model$N <- length(dims) - 1
    ae_model$dims <- dims
    ae_model$stacks <- list()
    ae_model$pt_dropout <- pt_dropout
    ae_model$ft_dropout <- ft_dropout
    ae_model$input_act <- input_act
    ae_model$internal_act <- internal_act
    ae_model$output_act <- output_act
    
    for (i in 1:ae_model$N) {
      if (i == 1) {
        decoder_act = input_act
        idropout = NULL
      } else {
        decoder_act = internal_act
        idropout = pt_dropout
      }
      
      if (i == ae_model$N) {
        encoder_act = output_act
        odropout = NULL
      } else {
        encoder_act = internal_act
        odropout = pt_dropout
      }
      
      istack <-
        make_stack(
          ctx, i, ae_model$data, dims[i], dims[i + 1],
          sparseness_penalty, idropout, odropout, encoder_act, decoder_act
        )
      
    }
    
  }