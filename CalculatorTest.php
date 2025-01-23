<?php
// tests/CalculatorTest.php

require_once __DIR__ . '/../src/Calculator.php';

use PHPUnit\Framework\TestCase;

class CalculatorTest extends TestCase
{
    private $calculator;

    protected function setUp(): void
    {
        $this->calculator = new Calculator();
    }

// =========================================================================
//  Test case berhasil     

    public function testAdd()
    {
        $this->assertEquals(5, $this->calculator->add(2, 3));
        $this->assertEquals(-1, $this->calculator->add(-2, 1));
    }

    public function testSubtract()
    {
        $this->assertEquals(1, $this->calculator->subtract(3, 2));
        $this->assertEquals(-3, $this->calculator->subtract(-1, 2));
    }

    public function testMultiply()
    {
        $this->assertEquals(6, $this->calculator->multiply(2, 3));
        $this->assertEquals(0, $this->calculator->multiply(0, 3));
    }

    public function testDivide()
    {
        $this->assertEquals(2, $this->calculator->divide(6, 3));
        $this->assertEquals(-2, $this->calculator->divide(-6, 3));
        $this->expectException(\InvalidArgumentException::class);
        $this->calculator->divide(5, 0);
    }

    // public function testDivideByZero()
    // {
    //     $this->expectException(\InvalidArgumentException::class);
    //     $this->calculator->divide(5, 0);
    // }

// ==========================================================================

// // // Test case gagal 
// public function testAddFailed()
// {
//     // Ini akan gagal karena hasil penambahan seharusnya 5
//     $this->assertEquals(6, $this->calculator->add(2, 3)); 

//     // Ini akan gagal karena hasil seharusnya -1
//     $this->assertEquals(0, $this->calculator->add(-2, 1)); 
// }

// // Test case gagal untuk method subtract()
// public function testSubtractFailed()
// {
//     // Ini akan gagal karena hasil pengurangan seharusnya 1
//     $this->assertEquals(2, $this->calculator->subtract(3, 2)); 

//     // Ini akan gagal karena hasil pengurangan seharusnya -3
//     $this->assertEquals(-2, $this->calculator->subtract(-1, 2)); 
// }

// // Test case gagal untuk method multiply()
// public function testMultiplyFailed()
// {
//     // Ini akan gagal karena hasil perkalian seharusnya 6
//     $this->assertEquals(5, $this->calculator->multiply(2, 3));

//     // Ini akan gagal karena hasil seharusnya 0
//     $this->assertEquals(1, $this->calculator->multiply(0, 3));
// }

// // Test case gagal untuk method divide()
// public function testDivideFailed()
// {
//     // Ini akan gagal karena hasil pembagian seharusnya 2
//     $this->assertEquals(3, $this->calculator->divide(6, 3)); 

//     // Ini akan gagal karena hasil seharusnya -2
//     $this->assertEquals(-3, $this->calculator->divide(-6, 3)); 
// }


 }
