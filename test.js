/**
 * Created by ARVIND on 10/11/2017.
 */
"use strict";

module("ttt");

test("new board is empty", function () {
    var b = ttt.newBoard();
    ok(ttt.isEmpty(b), "empty");
    for (var i = 0; i < 9; ++i) {
        strictEqual(ttt.getPiece(b, i), 0, "getPiece(" + i + ") is blank");
    }
    deepEqual(ttt.toArray(b), [0, 0, 0, 0, 0, 0, 0, 0, 0], "blank toArray");
    deepEqual(ttt.emptySquares(b), [0, 1, 2, 3, 4, 5, 6, 7, 8], "all squares empty");
    strictEqual(ttt.winner(b), 0, "no winner yet");
});

test("making moves is internally consistent", function () {
    var square = 4;
    var piece = ttt.X;
    var b = ttt.move(ttt.newBoard(), square, piece);
    ok(!ttt.isEmpty(b), "not empty");
    strictEqual(ttt.getPiece(b, square), piece, "getPiece returns correct piece");
    deepEqual(ttt.toArray(b), [0, 0, 0, 0, piece, 0, 0, 0, 0], "toArray has correct piece");
    deepEqual(ttt.emptySquares(b), [0, 1, 2, 3, 5, 6, 7, 8], "square isn't empty");
    strictEqual(ttt.winner(b), 0, "no winner yet");
});

test("win conditions", function () {
    var b;
    [ttt.X, ttt.O].forEach(function (piece) {
        for (var i = 0; i < 3; ++i) {
            b = ttt.newBoard();
            b = ttt.move(b, i * 3 + 0, piece);
            b = ttt.move(b, i * 3 + 1, piece);
            b = ttt.move(b, i * 3 + 2, piece);
            strictEqual(ttt.winner(b), piece, (piece === ttt.X ? "X" : "O") + " wins, horizontal " + i);

            b = ttt.newBoard();
            b = ttt.move(b, i + 0, piece);
            b = ttt.move(b, i + 3, piece);
            b = ttt.move(b, i + 6, piece);
            strictEqual(ttt.winner(b), piece, (piece === ttt.X ? "X" : "O") + " wins, vertical " + i);
        }

        b = ttt.newBoard();
        b = ttt.move(b, 0, piece);
        b = ttt.move(b, 4, piece);
        b = ttt.move(b, 8, piece);
        strictEqual(ttt.winner(b), piece, (piece === ttt.X ? "X" : "O") + " wins, diagonal 0");

        b = ttt.newBoard();
        b = ttt.move(b, 2, piece);
        b = ttt.move(b, 4, piece);
        b = ttt.move(b, 6, piece);
        strictEqual(ttt.winner(b), piece, (piece === ttt.X ? "X" : "O") + " wins, diagonal 1");
    });

    b = ttt.newBoard();
    b = ttt.move(b, 0, ttt.X);
    b = ttt.move(b, 1, ttt.O);
    b = ttt.move(b, 2, ttt.X);
    b = ttt.move(b, 3, ttt.X);
    b = ttt.move(b, 4, ttt.O);
    b = ttt.move(b, 5, ttt.O);
    b = ttt.move(b, 6, ttt.O);
    b = ttt.move(b, 7, ttt.X);
    b = ttt.move(b, 8, ttt.X);
    strictEqual(ttt.winner(b), ttt.TIE, "cat's game");
});

test("Game logic", function () {
    var g = new ttt.Game();
    strictEqual(g.board, ttt.newBoard(), "new Game has new board");
    strictEqual(g.turn, ttt.X, "X goes first");
    deepEqual(g.history, [], "no history yet");
    g.move(4);
    strictEqual(g.board, ttt.move(ttt.newBoard(), 4, ttt.X), "Game's board correct after one move");
    strictEqual(g.turn, ttt.O, "O goes second");
    deepEqual(g.history, [ttt.newBoard()], "history is empty board");
    g.move(0);
    strictEqual(g.turn, ttt.X, "X goes third");
    deepEqual(g.history, [ttt.newBoard(), ttt.move(ttt.newBoard(), 4, ttt.X)], "history updated");
    var g2 = g.clone();
    ok(g.equals(g2), "clone equal");
    deepEqual(g.history, g2.history, "clone history equal");
    g.undo();
    strictEqual(g.board, ttt.move(ttt.newBoard(), 4, ttt.X), "board undone");
    strictEqual(g.turn, ttt.O, "turn undone");
    deepEqual(g.history, [ttt.newBoard()], "history undone");
    ok(!g.equals(g2), "clone no longer equal");
});

module("Neural");

test("xor", function () {
    var n = new Neural.Net([2, 3, 1]);
    deepEqual(n.getSizes(), [2, 3, 1], "correct sizes");
    n.setWeights([[[1, 0.5, 0], [0, 0.5, 1]], [[1], [-2], [1]]]);
    deepEqual(n.getWeights(), [[[1, 0.5, 0], [0, 0.5, 1]], [[1], [-2], [1]], [[]]], "correct weights");
    deepEqual(n.run([0, 0]), [0], "0⊕0 = 0");
    n.reset();
    deepEqual(n.run([0, 1]), [1], "0⊕1 = 1");
    n.reset();
    deepEqual(n.run([1, 0]), [1], "1⊕0 = 1");
    n.reset();
    deepEqual(n.run([1, 1]), [0], "1⊕1 = 0");
});

test("cloned/exported/imported xor", function () {
    var n = new Neural.Net([2, 3, 1]);
    n.setWeights([[[1, 0.5, 0], [0, 0.5, 1]], [[1], [-2], [1]]]);
    var n2 = n.clone();
    n.setWeights([[[0, 0, 0], [0, 0, 0]], [[0], [0], [0]]]);
    var exp = n2.export();
    n2.setWeights([[[0, 0, 0], [0, 0, 0]], [[0], [0], [0]]]);
    var exportJson = JSON.stringify(exp);
    var n3 = Neural.Net.import(JSON.parse(exportJson));
    deepEqual(n3.getSizes(), [2, 3, 1], "correct sizes");
    deepEqual(n3.getWeights(), [[[1, 0.5, 0], [0, 0.5, 1]], [[1], [-2], [1]], [[]]], "correct weights");
    deepEqual(n3.run([0, 0]), [0], "0⊕0 = 0");
    n3.reset();
    deepEqual(n3.run([0, 1]), [1], "0⊕1 = 1");
    n3.reset();
    deepEqual(n3.run([1, 0]), [1], "1⊕0 = 1");
    n3.reset();
    deepEqual(n3.run([1, 1]), [0], "1⊕1 = 0");
});

test("nand", function () {
    var n = new Neural.Net([2, 1, 1]);
    deepEqual(n.getSizes(), [2, 1, 1], "correct sizes");
    n.setWeights([[[-1], [-1]], [[1]]]);
    n.setThresholds([[1, 1], [-1]]);
    deepEqual(n.getWeights(), [[[-1], [-1]], [[1]], [[]]], "correct weights");
    deepEqual(n.getThresholds(), [[1, 1], [-1], [undefined]], "correct thresholds");
    deepEqual(n.run([0, 0]), [1], "0↑0 = 1");
    n.reset();
    deepEqual(n.run([0, 1]), [1], "0↑1 = 1");
    n.reset();
    deepEqual(n.run([1, 0]), [1], "1↑0 = 1");
    n.reset();
    deepEqual(n.run([1, 1]), [0], "1↑1 = 0");
});

module("Ai");

test("basic Smart behavior", function () {
    var a = new Ai.Smart();
    var g = new ttt.Game();
    strictEqual(a.getMove(g), 4, "center first");
    g.move(0);
    g.move(3);
    g.move(1);
    strictEqual(a.getMove(g), 2, "blocks an immediate threat");
    g.move(4);
    strictEqual(a.getMove(g), 2, "goes for a win over blocking");
});

test("Smart vs. itself", function () {
    var a = new Ai.Smart();
    var g = new ttt.Game();
    var move = a.getMove(g);
    ok(move >= 0 && move < 9 && g.getPiece(move) === 0, "valid first move");
    g.move(move);
    move = a.getMove(g);
    ok(move >= 0 && move < 9 && g.getPiece(move) === 0, "valid second move");
    g.move(move);
    move = a.getMove(g);
    ok(move >= 0 && move < 9 && g.getPiece(move) === 0, "valid third move");
    g.move(move);
    move = a.getMove(g);
    ok(move >= 0 && move < 9 && g.getPiece(move) === 0, "valid fourth move");
    g.move(move);
    move = a.getMove(g);
    ok(move >= 0 && move < 9 && g.getPiece(move) === 0, "valid fifth move");
    g.move(move);
    move = a.getMove(g);
    ok(move >= 0 && move < 9 && g.getPiece(move) === 0, "valid sixth move");
    g.move(move);
    move = a.getMove(g);
    ok(move >= 0 && move < 9 && g.getPiece(move) === 0, "valid seventh move");
    g.move(move);
    move = a.getMove(g);
    ok(move >= 0 && move < 9 && g.getPiece(move) === 0, "valid eighth move");
    g.move(move);
    move = a.getMove(g);
    ok(move >= 0 && move < 9 && g.getPiece(move) === 0, "valid ninth move");
    g.move(move);
    strictEqual(g.winner(), ttt.TIE, "cat's game");
});

test("basic Neural behavior", function () {
    var n = new Neural.Net([18, 1]);
    n.setWeights([
        [[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]],
        [[1]]
    ]);
    n.setThresholds([[2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2], [1]]);
    var a = new Ai.Neural(n);
    var g = new ttt.Game();
    strictEqual(a.getMove(g), 4, "chooses highest scoring move");
});